# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drive a candidate ``jax_repo`` at grade time (standalone; no torch, no engine).

A :class:`Candidate` loads the candidate repo **inside a freshly spawned,
torch-blocked subprocess** and runs the candidate-side artifacts
and independence probes against the **frozen** inputs captured from the reference
(``golden/inputs.npz``). It mirrors the jax-path methods of the authoring
engine's ``runtime.py`` (forward/loss/grad/replay/generation + structural,
forward-purity, gradient-authenticity, determinism), but never imports torch and
never builds a torch collator: the batch is provided, not regenerated.

The static torch-import scan (:meth:`Candidate.purity_scan`) runs in the parent
process — it needs only the standard library.

Method coverage is complete for all three methods. :meth:`Candidate.loss_artifact`
and the gradient path dispatch on ``training_method`` for SFT, DPO, and PPO (the
numeric DPO/PPO formulas are vendored in :mod:`_formulas`); an unrecognised method
raises ``ValueError``. DPO additionally loads a frozen reference policy (the same
checkpoint, converted again) and PPO exercises its value head.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import _formulas as F
from . import purity, sandbox

# Marker env var propagated to child processes so any nested spawn also blocks.
SANDBOX_ENV_FLAG = "T2J_TORCH_SANDBOX"

# Generation input keys forwarded to model.generate (superset; missing keys skip).
_GENERATION_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "image_grid_thw",
    "video_grid_thw",
    "position_ids",
    "rope_deltas",
)


class CandidateError(RuntimeError):
    """Raised in the parent when the sandboxed candidate failed to produce artifacts."""


# Load-bearing training-config DEFAULTS the port must resolve identically
# (test_config_spec, config-DEFAULT parity / Flavor A). Both the grader-side freezer
# (run against torch_orig) and this probe resolve
# ``config.get_train_args({})`` -- the pure config.py DATACLASS DEFAULTS under an EMPTY
# override dict (NOT ``None``, which routes to ``sys.argv`` on both stacks), plus the light
# in-config post-processing (lora_alpha=lora_rank*2, template setdefault, use_ref_model
# derivation). Each field is read from whichever returned dataclass owns it, first owner wins;
# a field absent on this method's config (a PPO knob on an SFT task) is skipped on BOTH sides,
# so one broad list serves all three methods and only the applicable fields get frozen/checked.
#
# Scope: only fields that exist as a real dataclass default in BOTH stacks (AST-verified
# shared), so a torch-present/jax-absent field can never manufacture a false mismatch.
# Deliberately EXCLUDED: compute_dtype (torch resolves it to a torch.dtype object whose
# representation diverges from the JAX dtype; its parity is already covered numerically);
# output_dir / dataset / cache paths / io-cadence knobs (not load-bearing); and the
# canonical-grade-inert knobs val_size / num_train_epochs / do_eval (their dataclass defaults,
# not a spec check, drive the in-band grade).
_CONFIG_SPEC_FIELDS = (
    # core optimization (all methods)
    "learning_rate", "lr_scheduler_type", "weight_decay", "warmup_steps", "warmup_ratio",
    "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
    "max_grad_norm", "seed",
    # data / masking defaults the in-band trainer reads
    "template", "cutoff_len", "train_on_prompt", "max_samples",
    # finetuning / LoRA
    "finetuning_type", "lora_rank", "lora_alpha", "lora_dropout", "lora_target",
    # preference / DPO
    "pref_beta", "pref_loss", "pref_ftx", "simpo_gamma", "dpo_label_smoothing",
    "disable_dropout", "use_ref_model",
    # PPO (reward_model default now aligned to None across both stacks)
    "reward_model", "reward_model_type", "kl_coef", "ppo_epochs", "ppo_buffer_size",
    "ppo_target", "ppo_score_norm", "ppo_whiten_rewards", "vf_coef", "cliprange",
    "cliprange_value", "gamma", "lam",
    # generation (PPO rollout + eval)
    "max_new_tokens", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
)


def _config_json_safe(v: Any) -> Any:
    """Coerce a config field to a JSON-comparable value (enums -> their str, tuples -> lists)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_config_json_safe(x) for x in v]
    return str(v)


def _token_id_safe(v: Any) -> Any:
    """A special-token id may be an int, a list (multi-EOS tokenizers), or absent."""
    if v is None or isinstance(v, int):
        return v
    if isinstance(v, (list, tuple)):
        return [_token_id_safe(x) for x in v]
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


# =========================================================================== #
# In-sandbox runtime (torch-free reimplementation of the jax candidate path).
# =========================================================================== #
class _CandidateRuntime:
    """Built inside the torch-blocked subprocess; computes all candidate artifacts."""

    def __init__(self, payload: dict[str, Any]) -> None:
        import sys

        self.payload = payload
        self.training_method = payload["training_method"]
        self.seed = int(payload.get("seed", 42))
        self.loss_trajectory_steps = int(payload.get("loss_trajectory_steps", 3))
        self.generation_max_new_tokens = int(payload.get("generation_max_new_tokens", 16))
        # Reference's frozen generated sequence (prompt + greedy continuation), used by
        # the teacher-forced continuation-logit probe. None when the
        # golden froze no generation (DPO/PPO) -> the probe reports unsupported and its
        # test is deselected.
        _gen_seq = payload.get("generation_sequences")
        self.generation_sequences = None if _gen_seq is None else np.asarray(_gen_seq)
        # Per-variant frozen inputs for the input-space battery
        # ({variant: {input_ids, attention_mask, labels}}). None when the golden froze no
        # battery (multimodal / DPO+PPO of non-text families / older golden) -> the probe
        # reports unsupported and its tests are deselected.
        _battery = payload.get("battery_inputs")
        self.battery_inputs = (
            None if _battery is None
            else {name: {k: np.asarray(v) for k, v in inp.items()} for name, inp in _battery.items()}
        )
        self.compare_hidden_states = bool(payload.get("compare_hidden_states", False))
        self.repo_dir = str(Path(payload["repo_dir"]).resolve())
        self.jax_device = F.resolve_jax_device(payload.get("device"))

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        import importlib

        config_mod = importlib.import_module("config")
        self._config_mod = config_mod
        self.model_loader = importlib.import_module("model.loader")

        repo_config = dict(payload["repo_config"])
        args = config_mod.get_train_args(repo_config)
        self.model_args = args[0]
        self.data_args = args[1]
        self.training_args = args[2]
        self.finetuning_args = args[3]
        self.extra_args = args[4:]
        self.ignore_index = F.get_ignore_index(self.repo_dir)

        tok = self.model_loader.load_tokenizer(self.model_args)
        self.tokenizer = tok["tokenizer"] if isinstance(tok, dict) else tok
        self.processor = tok.get("processor") if isinstance(tok, dict) else None
        self._repo_config = repo_config

        self.model, self.params = self._load_model()
        if self.params is None:
            raise ValueError("candidate load_model did not expose a JAX params pytree")

        # DPO frozen reference policy (the same checkpoint, converted again) when the
        # candidate exposes load_ref_model and the flow needs one.
        self.ref_model, self.ref_params = self._load_ref_model()

        # Frozen inputs: numpy arrays from golden/inputs.npz -> jax arrays.
        self._np_batch = {k: np.asarray(v) for k, v in payload["inputs"].items()}
        self.batch = {k: F.to_jax_on_device(v, self.jax_device) for k, v in self._np_batch.items()}

        # Batch rebuilt via the candidate's OWN data pipeline; built
        # once and shared by candidate_batch_schema (keys+shape) and candidate_batch_arrays
        # (content), since run_all produces both as separate steps.
        self._candidate_batch_cache: dict[str, Any] | None = None

        # Lazily-built float32 upcast of self.params, shared across the run_all fp32
        # probes (the input-space battery's 4 fp32 forwards + fp32_forward). Built on first use
        # in _fp32_params_cached and dropped by _drop_fp32_params_cache the moment the
        # fp32 phase ends (run_all), so the 2x-checkpoint tree never lingers into the
        # memory-hungry trailing artifacts (gradient_authenticity). None between phases.
        self._fp32_params_cache: Any | None = None

    def _load_model(self) -> tuple[Any, Any]:
        import inspect

        # No weights are handed in: the candidate's own
        # model/loader.py::convert_checkpoint reads the HF checkpoint at
        # /data/{repo_id}, which model_args.model_name_or_path names.
        kwargs: dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "model_args": self.model_args,
            "finetuning_args": self.finetuning_args,
            "is_trainable": True,
        }
        if self.training_method == "ppo":
            # The value head is in no checkpoint, so nothing can infer it from the
            # weights: the caller asks for it, exactly as torch's load_model does.
            kwargs["add_valuehead"] = True
        sig = inspect.signature(self.model_loader.load_model)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepted = {k: v for k, v in kwargs.items() if has_var_kw or k in sig.parameters}
        loaded = self.model_loader.load_model(**accepted)
        # Contract: (module, params, frozen_prefixes) tuple, or a bundle exposing
        # .module/.params. The first tuple element must be the
        # apply-able flax module so the structural probe sees a native model.
        if isinstance(loaded, tuple):
            module = loaded[0]
            params = loaded[1] if len(loaded) > 1 else None
            return module, params
        module = getattr(loaded, "module", loaded)
        params = getattr(loaded, "params", None)
        return module, params

    def _load_ref_model(self) -> tuple[Any, Any]:
        """Frozen reference policy for DPO (same checkpoint); None if not applicable."""
        if self.training_method != "dpo":
            return None, None
        loader = getattr(self.model_loader, "load_ref_model", None)
        if not callable(loader):
            return None, None
        import inspect

        kwargs: dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "model_args": self.model_args,
            "finetuning_args": self.finetuning_args,
        }
        if self.extra_args:
            kwargs["dpo_args"] = self.extra_args[0]
        sig = inspect.signature(loader)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepted = {k: v for k, v in kwargs.items() if has_var_kw or k in sig.parameters}
        loaded = loader(**accepted)
        if loaded is None:
            return None, None
        if isinstance(loaded, tuple):
            return loaded[0], (loaded[1] if len(loaded) > 1 else None)
        return getattr(loaded, "module", loaded), getattr(loaded, "params", None)

    # --- batch helpers ---------------------------------------------------- #
    def _forward_inputs_of(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in batch.items()
            if not (self.training_method == "ppo" and k == "labels")
        }

    def _forward_inputs(self) -> dict[str, Any]:
        return self._forward_inputs_of(self.batch)

    def _apply_on(
        self, params: Any, batch: dict[str, Any], *, output_hidden_states: bool = False
    ) -> Any:
        return F.call_model_with_control_kwargs(
            self.model.apply,
            {"params": params},
            model_inputs=self._forward_inputs_of(batch),
            return_dict=True,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )

    def _apply(self, params: Any, *, output_hidden_states: bool = False) -> Any:
        return self._apply_on(params, self.batch, output_hidden_states=output_hidden_states)

    # --- trainable partition (LoRA) --------------------------------------- #
    #: Flat-name markers for a LoRA adapter leaf. The same convention
    #: test_gradient_per_param already requires of a LoRA candidate, so this adds no
    #: new naming obligation -- it just spends the obligation twice.
    _LORA_MARKERS = ("lora_a", "lora_b")
    #: ...and PPO's value head, which LoRA does not freeze: it is not part of the
    #: checkpoint at all, it is created fresh and zero-initialised by the trainer, and
    #: golden/gradients.json for every PPO instance holds exactly one non-zero
    #: leaf -- ``v_head.weight``. Leaving it out would make the candidate's grad_norm
    #: identically zero for those instances.
    _VALUE_HEAD_MARKER = "v_head"

    def _trainable_keys(self, params: Any) -> list[str] | None:
        """Flat names of the leaves that carry a gradient, or ``None`` when that is all
        of them (full finetuning).

        LoRA freezes the base checkpoint and trains only the low-rank adapters. The
        reference does exactly that, ``golden/gradients.json`` holds only the trained
        leaves (LoRA adapters, plus PPO's value head) for every instance -- all 45 grade
        under ``finetuning_type=lora`` -- and the LoRA branches of
        test_gradient_per_param / test_loss_trajectory read only those. Differentiating
        the frozen base anyway is not merely waste: reverse mode has to hold a gradient
        buffer the size of the checkpoint, and the SGD replay in
        loss_trajectory_artifact then *moves* weights the reference never moves -- so
        the full-tree gradient is both more expensive and less faithful. On the 7B
        instances the buffer is what put the artifact over the single GPU task.toml
        gives the verifier.
        """
        if str(self._repo_config.get("finetuning_type", "full")).lower() != "lora":
            return None
        names = list(F.flatten_tree_raw(params))
        markers = self._LORA_MARKERS
        if self.training_method == "ppo":
            markers = markers + (self._VALUE_HEAD_MARKER,)
        keep = [n for n in names if any(m in n.lower() for m in markers)]
        if not keep or len(keep) == len(names):
            # No adapter leaf under the agreed naming, or every leaf is one. Say nothing
            # and differentiate the whole tree: test_gradient_per_param's LoRA branch is
            # the test that owns "this candidate has no adapters", and it reports that
            # far better than a silently empty gradient would.
            return None
        return keep

    def _value_and_grad_flat(self, loss_fn, params: Any, keep: list[str] | None):
        """``jax.value_and_grad(loss_fn)(params)`` as ``(loss, grads_tree, {name: grad})``.

        With ``keep`` the differentiation is restricted to those leaves and everything
        else is closed over as a constant, so reverse mode never allocates a cotangent
        for the frozen base; ``grads_tree`` is then ``None``, because there is no
        full-tree gradient to hand back. Without it the whole tree is differentiated and
        ``grads_tree`` is the gradient exactly as JAX produced it -- same container type
        as ``params``, so a caller can still ``tree_map`` the two together.
        """
        import jax

        if keep is None:
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            return loss_value, grads, F.flatten_tree_raw(grads)
        flat = F.flatten_tree_raw(params)
        trainable = [flat[name] for name in keep]

        def _on(sub: list[Any]) -> Any:
            merged = dict(flat)
            merged.update(zip(keep, sub))
            return loss_fn(_unflatten_raw(merged))

        loss_value, grads = jax.value_and_grad(_on)(trainable)
        return loss_value, None, dict(zip(keep, grads))

    # --- spec / structural ------------------------------------------------ #
    def param_spec(self) -> dict[str, dict[str, Any]]:
        flat = F.flatten_tree_raw(self.params)
        spec: dict[str, dict[str, Any]] = {}
        for key, leaf in flat.items():
            shape = [int(d) for d in getattr(leaf, "shape", ())]
            spec[key] = {"shape": shape, "dtype": str(getattr(leaf, "dtype", "?"))}
        return spec

    def param_norms(self) -> dict[str, dict[str, Any]]:
        """Layout-invariant summary of the LOADED weights.

        param_spec above says the tree has the right *shape*; this says it holds the
        right *numbers* — which is the half that matters now that the candidate converts
        the HF checkpoint itself. ``grad_stats`` is the same transposition- and
        permutation-invariant float64 reduction the per-parameter gradient gate uses, and
        ``golden/param_norms.json`` was produced by running it over the reference's own
        converted weights, so the two sides differ only by accumulation order."""
        flat = F.flatten_tree_raw(self.params)
        return {key: F.grad_stats(_to_np(leaf)) for key, leaf in flat.items()}

    def structural_probe(self) -> dict[str, Any]:
        import jax

        info: dict[str, Any] = {"has_pt_model": getattr(self.model, "pt_model", None) is not None}
        info["module_type"] = f"{type(self.model).__module__}.{type(self.model).__name__}"
        info["module_is_flax"] = F.is_flax_module(self.model)
        leaves = jax.tree_util.tree_leaves(self.params)
        jax_leaves = sum(1 for leaf in leaves if isinstance(leaf, jax.Array))
        dtypes = sorted({str(getattr(leaf, "dtype", "?")) for leaf in leaves})
        param_count = 0
        for leaf in leaves:
            shape = getattr(leaf, "shape", None)
            if shape is not None:
                param_count += int(np.prod(shape)) if shape else 1
        info.update(
            {
                "is_pytree": True,
                "num_leaves": len(leaves),
                "jax_array_leaves": jax_leaves,
                "all_jax_arrays": jax_leaves == len(leaves) and len(leaves) > 0,
                "leaf_dtypes": dtypes,
                "param_count": param_count,
            }
        )
        return info

    def forward_purity_probe(self, min_primitives: int = 8) -> dict[str, Any]:
        import jax

        from .purity import count_jaxpr_primitives, jaxpr_uses_callbacks

        def forward(params: Any) -> Any:
            out = self._apply(params)
            # Reduce the forward's return to its array/tracer leaves before
            # tracing, so make_jaxpr accepts *any* return container -- a bare
            # array, tuple, dict, NamedTuple/dataclass, or a non-pytree HF-style
            # ModelOutput. A non-pytree return is not a purity violation; the
            # property under test (real XLA ops, no host callbacks) is a
            # property of the traced computation, not of the output container.
            # Every primitive of self._apply is recorded during tracing
            # regardless of which leaves are returned, so the >= min_primitives
            # and callback signals are preserved.
            leaves = [
                x
                for x in jax.tree_util.tree_leaves(out)
                if hasattr(x, "shape") and hasattr(x, "dtype")
            ]
            if not leaves:
                # Opaque (non-pytree) container: pull known output fields by name.
                for field in ("logits", "loss", "last_hidden_state", "hidden_states", "values"):
                    value = F.extract_output_field(out, field)
                    leaves.extend(
                        x
                        for x in jax.tree_util.tree_leaves(value)
                        if hasattr(x, "shape") and hasattr(x, "dtype")
                    )
            return leaves

        try:
            if self.jax_device is None:
                closed = jax.make_jaxpr(forward)(self.params)
            else:
                with jax.default_device(self.jax_device):
                    closed = jax.make_jaxpr(forward)(self.params)
        except Exception as exc:  # surfaced as a failed check
            return {"supported": False, "reason": f"make_jaxpr failed: {exc}"}
        num_primitives = count_jaxpr_primitives(closed)
        callbacks = jaxpr_uses_callbacks(closed)
        return {
            "supported": True,
            "num_primitives": num_primitives,
            "min_primitives": min_primitives,
            "callbacks": callbacks,
            "traced": num_primitives >= min_primitives and not callbacks,
        }

    # --- forward / loss --------------------------------------------------- #
    def forward_outputs(self, *, output_hidden_states: bool | None = None) -> dict[str, Any]:
        if output_hidden_states is None:
            output_hidden_states = self.compare_hidden_states
        output = self._apply(self.params, output_hidden_states=output_hidden_states)
        logits = F.extract_output_field(output, "logits")
        loss = F.extract_output_field(output, "loss")
        hidden_states = F.extract_output_field(output, "hidden_states")
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
            if len(output) > 1:
                loss = output[1]
            if len(output) > 2:
                hidden_states = output[2]
        result: dict[str, Any] = {
            "logits": _to_np(logits),
            "loss": None if loss is None else float(_to_np(loss).reshape(())),
        }
        # Native-loss fallback: the reference freezes forward.npz["loss"] as the shifted
        # causal CE (its _torch_causal_loss, not the module's internal loss), so an SFT/DPO
        # golden always carries one. A flax module that returns logits only leaves loss=None,
        # which would fail the forward-loss parity check for a correct port; recompute the same
        # shifted causal CE from our own logits + labels so the check has a value to compare.
        # PPO has no causal loss and freezes none (its check is deselected), so it is excluded.
        # Mirrors fp32_forward_outputs and loss_artifact's SFT fallback.
        if result["loss"] is None and result["logits"] is not None and self.training_method != "ppo":
            labels = self._np_batch.get("labels", self._np_batch["input_ids"])
            result["loss"] = float(F.causal_cross_entropy(result["logits"], labels, self.ignore_index))
        if hidden_states is not None and self.compare_hidden_states:
            result["hidden_states"] = [_to_np(item) for item in hidden_states]
        values = F.extract_output_field(output, "values")
        if values is not None and hasattr(values, "shape"):
            result["values"] = _to_np(values)
        return result

    def _ref_logps_np(self, labels: np.ndarray) -> np.ndarray | None:
        """Reference-policy per-sequence log-probs for DPO (numpy)."""
        if self.ref_model is None or self.ref_params is None:
            return None
        ref_out = F.call_model_with_control_kwargs(
            self.ref_model.apply,
            {"params": self.ref_params},
            model_inputs={k: v for k, v in self.batch.items() if k != "labels"},
            return_dict=True,
            use_cache=False,
        )
        ref_logits = _to_np(F.extract_output_field(ref_out, "logits"))
        return F.batch_logps(ref_logits, labels, self.ignore_index)[0]

    def loss_artifact(self) -> dict[str, Any]:
        outputs = self.forward_outputs(output_hidden_states=False)
        labels = self._np_batch.get("labels", self._np_batch["input_ids"])
        logits = outputs["logits"]
        if self.training_method == "sft":
            loss = outputs["loss"]
            if loss is None:
                loss = F.causal_cross_entropy(logits, labels, self.ignore_index)
            return {"loss": float(loss)}
        if self.training_method == "dpo":
            policy_logps, valid_len = F.batch_logps(logits, labels, self.ignore_index)
            ref_logps = self._ref_logps_np(labels)
            dpo = self.extra_args[0] if self.extra_args else None
            loss, details = F.dpo_loss(
                policy_logps,
                ref_logps,
                beta=float(getattr(dpo, "pref_beta", 0.1)),
                loss_type=str(getattr(dpo, "pref_loss", "sigmoid")),
                label_smoothing=float(getattr(dpo, "dpo_label_smoothing", 0.0)),
                simpo_gamma=float(getattr(dpo, "simpo_gamma", 0.5)),
            )
            return {
                "loss": loss,
                "policy_logps": policy_logps.tolist(),
                "valid_length": valid_len.tolist(),
                "reference_logps": None if ref_logps is None else ref_logps.tolist(),
                **details,
            }
        if self.training_method == "ppo":
            values = outputs.get("values")
            if values is None:
                raise ValueError("PPO candidate forward must return a `values` field.")
            labels_t = labels[:, 1:].astype(np.int64, copy=False)
            mask = self._np_batch.get("attention_mask", np.ones_like(labels))[:, 1:].astype(np.float32)
            logits_t, values_t, labels_t, mask = F.normalize_ppo_time_axis(logits, values, labels_t, mask)
            rewards = ((labels_t % 17).astype(np.float32) / 17.0) * mask
            losses = F.ppo_losses(logits_t, values_t, labels_t, mask, self.ignore_index, rewards)
            return {
                "loss": float(losses["policy_loss"] + losses["value_loss"]),
                "policy_loss": float(losses["policy_loss"]),
                "value_loss": float(losses["value_loss"]),
                "advantages": losses["advantages"].tolist(),
                "returns": losses["returns"].tolist(),
                "token_logprobs": losses["token_logprobs"].tolist(),
            }
        raise ValueError(f"Unsupported training method: {self.training_method!r}")

    # --- candidate-own data pipeline (batch schema) ---------- #
    def _build_candidate_batch(self) -> dict[str, Any]:
        """Build a batch via the candidate's OWN framework-neutral data pipeline.

        Mirrors the reference data path (template -> dataset processor -> collator
        -> first batch) using the candidate's data modules and returns the built batch
        as ``{"supported": True, "batch": {key: np.ndarray}}``. Returns
        ``{"supported": False, ...}`` when the candidate does not expose a data pipeline
        (e.g. a minimal weights-only reference that is fed frozen inputs). Built once and
        cached, then shared by ``candidate_batch_schema`` (keys+shape) and
        ``candidate_batch_arrays`` (content), which run_all produces as separate steps.
        """
        if self._candidate_batch_cache is not None:
            return self._candidate_batch_cache
        result = self._build_candidate_batch_uncached()
        self._candidate_batch_cache = result
        return result

    def _build_candidate_batch_uncached(self) -> dict[str, Any]:
        import importlib
        import inspect

        def _call_supported(fn, **kwargs):
            sig = inspect.signature(fn)
            has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            return fn(**{k: v for k, v in kwargs.items() if has_var or k in sig.parameters})

        eval_examples = self._repo_config.get("eval_examples")
        if not eval_examples:
            return {"supported": False, "reason": "no eval_examples in config"}
        try:
            template_mod = importlib.import_module("data.template")
            builder = getattr(template_mod, "get_template_and_fix_tokenizer", None)
            template = builder(self.tokenizer, self.data_args) if builder else None

            processor_mod = importlib.import_module("data.processor")
            proc_name = {"sft": "SupervisedDatasetProcessor", "dpo": "PairwiseDatasetProcessor",
                         "ppo": "UnsupervisedDatasetProcessor"}[self.training_method]
            processor_cls = getattr(processor_mod, proc_name, None)
            if processor_cls is None:
                return {"supported": False, "reason": f"no {proc_name} in data.processor"}
            proc = processor_cls(template=template, tokenizer=self.tokenizer, processor=self.processor, data_args=self.data_args)
            batched = {key: [ex.get(key) for ex in eval_examples] for key in eval_examples[0]}
            # t2j vision-numeric gate: decode an inlined ``data:`` image URI to a PIL.Image
            # upstream of the STOCK (agent-ported) data pipeline, so the candidate rebuilds the
            # golden's real image without any mm_plugin ``data:`` branch (the port stays stock).
            # PIL.Image is accepted unmodified by every family (_regularize_images fall-through /
            # get_mm_inputs isinstance(Image) branch). Byte-mirrors the freeze-time decode
            # applied when the golden was frozen.
            def _t2j_decode_media(_v):
                if isinstance(_v, str) and _v.startswith("data:"):
                    import base64 as _b64, io as _io
                    from PIL import Image as _PILImage
                    return _PILImage.open(_io.BytesIO(_b64.b64decode(_v.split(",", 1)[1])))
                if isinstance(_v, list):
                    return [_t2j_decode_media(_x) for _x in _v]
                return _v
            for _mkey in ("_images",):
                if _mkey in batched:
                    batched[_mkey] = [_t2j_decode_media(_c) for _c in batched[_mkey]]
            processed = proc.preprocess_dataset(batched)
            size = len(next(iter(processed.values())))
            dataset = [{k: v[i] for k, v in processed.items()} for i in range(size)]

            collator_mod = importlib.import_module("data.collator")
            cls_name = {"sft": "SFTDataCollatorWith4DAttentionMask", "dpo": "PairwiseDataCollatorWithPadding",
                        "ppo": "MultiModalDataCollatorForSeq2Seq"}[self.training_method]
            collator_cls = getattr(collator_mod, cls_name, None) or getattr(collator_mod, "MultiModalDataCollatorForSeq2Seq", None)
            if collator_cls is None:
                return {"supported": False, "reason": f"no collator class in data.collator"}
            # ``pad_to_multiple_of`` is NOT a collator default on either side -- both the
            # torch and the JAX collator default it to None and every train.py passes it
            # in from the training args (``getattr(training_args, "pad_to_multiple_of",
            # None) if training_args.do_train else None``). Building the collator without
            # it therefore grades a padding policy the reference never runs: nine
            # instances (gemma3 / llava_next / qwen3_vl x sft/dpo/ppo) set it to 8, and
            # their golden batches are padded up accordingly -- 85 -> 88, 281 -> 288.
            # Mirror train.py exactly rather than hard-coding a multiple.
            pad_to_multiple_of = getattr(self.training_args, "pad_to_multiple_of", None)
            if not getattr(self.training_args, "do_train", True):
                pad_to_multiple_of = None
            collator = _call_supported(
                collator_cls, template=template, model=self.model, tokenizer=self.tokenizer,
                processor=self.processor, label_pad_token_id=self.ignore_index,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            n = min(int(self._np_batch["input_ids"].shape[0]), len(dataset)) if "input_ids" in self._np_batch else len(dataset)
            batch = collator([dataset[i] for i in range(max(n, 1))])
            if hasattr(batch, "data") and isinstance(batch.data, dict):
                batch = dict(batch.data)
            # Materialise to numpy (bf16 -> fp32) so the batch pickles across the worker pipe
            # and both consumers see a framework-neutral array; integer keys keep their kind.
            arrays = {k: np.asarray(_to_np(v)) for k, v in dict(batch).items()}
            return {"supported": True, "batch": arrays}
        except Exception as exc:  # any missing piece -> unsupported, skip in the test
            return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}

    def candidate_batch_schema(self) -> dict[str, Any]:
        """Report the schema (keys -> shape/dtype) of the candidate's own rebuilt batch.

        Structural half of the data-pipeline grade; ``candidate_batch_arrays`` /
        ``test_data_pipeline`` grade the content. Returns
        ``{"supported": False, ...}`` when the candidate exposes no data pipeline.
        """
        built = self._build_candidate_batch()
        if not built.get("supported"):
            return built
        schema = {k: {"shape": [int(d) for d in v.shape], "dtype": str(v.dtype)}
                  for k, v in built["batch"].items()}
        return {"supported": True, "schema": schema}

    def candidate_batch_arrays(self) -> dict[str, Any]:
        """Return the actual arrays of the candidate's own rebuilt batch.

        Content half of the data-pipeline grade: ``test_data_pipeline`` compares these to
        the frozen ``golden/inputs.npz`` key-by-key (integer/token keys byte-exact, float
        keys best-effort), so a wrong chat template, a mis-masked label span, or a DPO
        collator that swaps chosen/rejected -- all shape-preserving, so invisible to
        ``test_batch_schema`` -- fails here. ``{"supported": False, ...}`` when the
        candidate exposes no runnable data pipeline.
        """
        built = self._build_candidate_batch()
        if not built.get("supported"):
            return built
        return {"supported": True, "arrays": built["batch"]}

    # --- gradient (value_and_grad over candidate apply + loss) ------------ #
    def _scalar_loss_fn_for(self, batch: dict[str, Any]):
        """The step-0/numeric scalar loss, closed over an arbitrary ``batch``.

        SFT is the model's own causal loss; DPO is the SIMPLIFIED preference-only term
        (matching the numeric ``loss_artifact`` golden -- odds-ratio / sigmoid margin, no
        length-normalised NLL); PPO is the inline single-forward GAE surrogate. This is
        the loss used by ``gradient_artifact``, ``loss_trajectory_artifact`` and the
        optimizer-signature probe. The FULL DPO trainer objective the ``train_trajectory``
        golden was frozen from lives in :func:`_dpo_trainer_loss_fn_for`."""
        import jax
        import jax.numpy as jnp
        import jax.scipy.special

        method = self.training_method
        labels = batch["labels"]
        ignore = self.ignore_index

        def scalar_loss(params: Any) -> Any:
            output = self._apply_on(params, batch)
            logits = F.extract_output_field(output, "logits")
            loss = F.extract_output_field(output, "loss")
            if logits is None and isinstance(output, tuple) and output:
                logits = output[0]
                if len(output) > 1:
                    loss = output[1]
            if method == "sft":
                return loss if loss is not None else F.jax_causal_loss(logits, labels, ignore)
            if method == "dpo":
                policy_logps, _ = F.jax_batch_logps(logits, labels, ignore)
                ref_logps = None
                if self.ref_model is not None and self.ref_params is not None:
                    ref_out = F.call_model_with_control_kwargs(
                        self.ref_model.apply,
                        {"params": self.ref_params},
                        model_inputs={k: v for k, v in batch.items() if k != "labels"},
                        return_dict=True,
                        use_cache=False,
                    )
                    ref_logps, _ = F.jax_batch_logps(F.extract_output_field(ref_out, "logits"), labels, ignore)
                    ref_logps = jax.lax.stop_gradient(ref_logps)
                dpo = self.extra_args[0] if self.extra_args else None
                beta = float(getattr(dpo, "pref_beta", 0.1))
                loss_type = str(getattr(dpo, "pref_loss", "sigmoid"))
                ls = float(getattr(dpo, "dpo_label_smoothing", 0.0))
                gamma = float(getattr(dpo, "simpo_gamma", 0.5))
                chosen_pi, rejected_pi = jnp.split(policy_logps, 2, axis=0)
                pi_logratios = chosen_pi - rejected_pi
                if ref_logps is None:
                    ref_logratios = jnp.zeros_like(pi_logratios)
                else:
                    chosen_ref, rejected_ref = jnp.split(ref_logps, 2, axis=0)
                    ref_logratios = chosen_ref - rejected_ref
                margin = pi_logratios - ref_logratios
                if loss_type == "sigmoid":
                    return -((1.0 - ls) * jax.nn.log_sigmoid(beta * margin) + ls * jax.nn.log_sigmoid(-beta * margin)).mean()
                if loss_type == "orpo":
                    return -jax.nn.log_sigmoid(beta * pi_logratios).mean()
                if loss_type == "simpo":
                    return -jax.nn.log_sigmoid(beta * (pi_logratios - gamma)).mean()
                raise ValueError(f"Unsupported DPO loss type: {loss_type}")
            if method == "ppo":
                values = F.extract_output_field(output, "values")
                if values is None or not hasattr(values, "shape"):
                    raise ValueError("PPO candidate forward must return a `values` field.")
                lab = labels[:, 1:].astype(jnp.int32)
                mask = batch.get("attention_mask", jnp.ones_like(labels))[:, 1:].astype(jnp.float32)
                lg, val, lab, mask = F.normalize_ppo_time_axis(
                    jnp.asarray(logits).astype(jnp.float32), jnp.asarray(values).astype(jnp.float32), lab, mask
                )
                safe_mask = mask * (lab != ignore).astype(jnp.float32)
                safe_labels = jnp.where(lab == ignore, 0, lab)
                rewards = ((safe_labels % 17).astype(jnp.float32) / 17.0) * safe_mask
                advantages = jnp.zeros_like(rewards)
                last = jnp.zeros(rewards.shape[0], dtype=jnp.float32)
                for ts in range(rewards.shape[1] - 1, -1, -1):
                    nxt = val[:, ts + 1] if ts < rewards.shape[1] - 1 else jnp.zeros_like(val[:, 0])
                    delta = rewards[:, ts] + nxt * safe_mask[:, ts] - val[:, ts]
                    last = delta + 0.95 * last * safe_mask[:, ts]
                    advantages = advantages.at[:, ts].set(last)
                returns = advantages + val
                policy_loss = ((-advantages) * safe_mask).sum() / jnp.maximum(safe_mask.sum(), 1.0)
                value_loss = 0.5 * (((val - returns) ** 2) * safe_mask).sum() / jnp.maximum(safe_mask.sum(), 1.0)
                return policy_loss + value_loss
            raise ValueError(f"Unsupported training method: {method!r}")

        return scalar_loss

    def _scalar_loss_fn(self):
        return self._scalar_loss_fn_for(self.batch)

    def _dpo_trainer_loss_fn_for(self, batch: dict[str, Any], ref_params: Any = None):
        """The FULL DPO-trainer objective closed over ``batch`` (differentiable).

        Reconstructs the reference ``DPOTrainer`` per-step loss from the documented forward
        contract alone: one concatenated chosen+rejected forward through ``module.apply``,
        then :func:`_formulas.dpo_trainer_loss` (length-normalised NLL + odds-ratio/sigmoid
        + ``ftx_gamma`` SFT term). This -- not the simplified preference term in
        :func:`_scalar_loss_fn_for` -- is what the ``train_trajectory`` golden was frozen
        from, so the reconstructed training curve reproduces it. For the reference-based
        sigmoid loss a frozen reference forward (``ref_params``) supplies the baseline
        log-probs; ORPO/SimPO are reference-free."""
        import jax

        labels = batch["labels"]
        ignore = self.ignore_index
        dpo = self.extra_args[0] if self.extra_args else None
        beta = float(getattr(dpo, "pref_beta", 0.1))
        loss_type = str(getattr(dpo, "pref_loss", "sigmoid"))
        ls = float(getattr(dpo, "dpo_label_smoothing", 0.0))
        gamma = float(getattr(dpo, "simpo_gamma", 0.5))
        ftx = float(getattr(dpo, "pref_ftx", 0.0))
        use_ref = loss_type not in ("orpo", "simpo")

        ref_all_logits = None
        if use_ref:
            ref_src_model = self.ref_model if self.ref_model is not None else self.model
            ref_src_params = ref_params if ref_params is not None else self.ref_params
            if ref_src_params is None:
                raise ValueError("reference-based DPO reconstruction requires reference params")
            ref_out = F.call_model_with_control_kwargs(
                ref_src_model.apply,
                {"params": ref_src_params},
                model_inputs=self._forward_inputs_of(batch),
                return_dict=True,
                use_cache=False,
            )
            ref_all_logits = jax.lax.stop_gradient(F.extract_output_field(ref_out, "logits"))

        def trainer_loss(params: Any) -> Any:
            output = self._apply_on(params, batch)
            logits = F.extract_output_field(output, "logits")
            if logits is None and isinstance(output, tuple) and output:
                logits = output[0]
            return F.dpo_trainer_loss(
                logits,
                labels,
                loss_type=loss_type,
                beta=beta,
                ftx_gamma=ftx,
                label_smoothing=ls,
                simpo_gamma=gamma,
                ignore_index=ignore,
                use_ref_model=use_ref,
                ref_all_logits=ref_all_logits,
            )

        return trainer_loss

    def gradient_artifact(self) -> dict[str, Any]:
        import jax

        loss_fn = self._scalar_loss_fn()
        # LoRA: only the adapters are trained, so only the adapters are differentiated
        # (see _trainable_keys). Full finetuning differentiates the whole tree.
        keep = self._trainable_keys(self.params)
        if self.jax_device is None:
            loss_value, _tree, raw = self._value_and_grad_flat(loss_fn, self.params, keep)
        else:
            with jax.default_device(self.jax_device):
                loss_value, _tree, raw = self._value_and_grad_flat(loss_fn, self.params, keep)
        flat = {k: _to_np(v) for k, v in raw.items()}
        import math as _math

        grad_norm_sq = sum(float(np.square(v.astype(np.float32)).sum()) for v in flat.values())
        return {
            "loss": float(np.asarray(loss_value)),
            "grad_norm": _math.sqrt(grad_norm_sq),
            # Per-parameter layout-invariant summary keyed by the candidate's pytree
            # path; test_gradient_per_param maps these to the torch reference by
            # canonical name.
            "grads": {k: F.grad_stats(v) for k, v in flat.items()},
        }

    # Cost model for the probe's float64 footprint, in bytes per parameter. Every
    # parameter reverse mode backpropagates THROUGH costs one float64 conversion
    # that is kept alive as a residual...
    _FD_BYTES_PER_CONVERTED_PARAM = 8
    # ...and each parameter actually perturbed costs, on top of that, its
    # cotangent and one perturbed copy (the two evaluated one at a time).
    _FD_BYTES_PER_SUPPORT_PARAM = 2 * 8

    def _fd_probe_capacity(self) -> int:
        """Bytes the authenticity probe may spend on float64 parameter copies.

        Read from the live allocator, because the answer differs by two orders of
        magnitude across the suite: the params are already resident (twice over for
        DPO, which also holds a frozen reference policy), and what is left on the
        one GPU the verifier is given (``task.toml``) is what the probe can use.
        """
        import jax

        free = 0
        try:
            device = self.jax_device or jax.local_devices()[0]
            stats = device.memory_stats() or {}
            free = int(stats.get("bytes_limit") or 0) - int(stats.get("bytes_in_use") or 0)
        except Exception:  # noqa: BLE001 - CPU backends expose no allocator stats
            free = 0
        if free <= 0:  # CPU: the params live in host memory, so ask the host
            try:
                with open("/proc/meminfo", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("MemAvailable:"):
                            free = int(line.split()[1]) * 1024
                            break
            except OSError:
                free = 0
        if free <= 0:
            free = 4 << 30
        # Keep a quarter back for what a per-parameter model cannot count: the
        # float64 temporaries XLA materialises when it upcasts and transposes a bf16
        # weight for a matmul, the probe's own activations, and fragmentation under
        # PREALLOCATE=false. The shortfall scales with the largest single leaf (a
        # large embedding costs several GiB the moment it is upcast), which is why
        # the reserve is a fraction of what is free rather than a constant.
        return max(free - max(free // 4, 4 << 30), 64 << 20)

    def _fd_leaf_span(
        self, leaves: list[Any], treedef: Any, loss_fn: Any
    ) -> tuple[list[int], list[int]] | None:
        """For each parameter leaf, the FIRST and LAST loss equation that reads it.

        Both ends matter, and for different reasons. The first read is the leaf's
        depth: perturbing it forces float64 from there on, so the probe wants the
        deepest leaves it can afford. The last read decides the bill: reverse mode
        keeps a float64 conversion of every bf16 weight it backpropagates through,
        so a leaf is charged for exactly when something it feeds is float64 — which
        is when its LAST read comes at or after the shallowest perturbed leaf. The
        distinction is not academic: a tied embedding table is read shallow (the
        token lookup) and deep (the logit matmul), and ranking it by its first read
        alone hides a 4.6 GiB conversion that reverse mode will make anyway.

        Equations are in trace order, so the indices are exactly these two ends, and
        they come from the candidate's own graph rather than from parameter names,
        which the agent chooses freely. Leaves the loss never reads get ``-1``: they
        have no gradient to check, and an eager ``astype(float64)`` would still bill
        them, so the caller drops them rather than spending its budget there.

        ``None`` when the candidate's loss cannot be traced abstractly.
        """
        import jax

        def _traced(sub: list[Any]) -> Any:
            return loss_fn(jax.tree_util.tree_unflatten(treedef, sub))

        try:
            jaxpr = jax.make_jaxpr(_traced)(leaves).jaxpr
        except Exception:  # noqa: BLE001 - an untraceable repo just loses the ordering
            return None
        if len(jaxpr.invars) != len(leaves):
            return None
        # Keyed by identity: a jaxpr's invars are Var objects, and eqn.invars also
        # holds Literals, which are not reliably hashable across jax versions.
        position = {id(var): index for index, var in enumerate(jaxpr.invars)}
        first = [-1] * len(leaves)
        last = [-1] * len(leaves)
        for eqn_index, eqn in enumerate(jaxpr.eqns):
            for var in eqn.invars:
                index = position.get(id(var))
                if index is None:
                    continue
                if first[index] < 0:
                    first[index] = eqn_index
                last[index] = eqn_index
        return first, last

    def native_gradient_authenticity(self, epsilon: float = 1e-4) -> dict[str, Any]:
        import jax
        import jax.numpy as jnp

        # Finite differences are meaningless on bf16 params (a small step rounds
        # away; a visible step is nonlinear), so the probe perturbs a float64 copy
        # of the parameters — x64 is enabled in the sandbox worker and the modules
        # promote, so every op downstream of a perturbed leaf runs in float64.
        # Probe SEVERAL directions: the gradient over the probe's
        # whole support (analytic = ||g_S||) plus that gradient restricted to two
        # disjoint halves of the support. A random direction projects to ~0 in a
        # ~5e8-dim space, so each direction stays aligned with g on its support —
        # together they catch a candidate that gets only SOME parameters' grads
        # right (a single full-support probe can be dominated by a correct majority).
        #
        # The support is capped by free device memory: three float64 copies of a 7B
        # tree are 174 GiB and cannot coexist on the single GPU the verifier is given
        # (task.toml), so above the cap the probe perturbs the DEEPEST leaves it can
        # afford and reports its coverage. Deepest, not a spread sample, because a
        # perturbed leaf drags everything after it into float64 whether it is being
        # perturbed or not, so a tail support is the one shape whose cost is only the
        # tail: ahead of it the two evaluations are bit-identical and the bf16
        # rounding cancels exactly. Leaves the loss never reads are dropped first —
        # they have no gradient to check, and the eager float64 copy would bill them
        # anyway (that alone is most of a Qwen2-Omni: probe the audio and talker
        # towers on a text batch and ||g_S|| comes back 0, which is not a test).
        # Models that fit keep full coverage — then the support IS the whole tree and
        # probes 1/2 are the even/odd leaf split, as before.
        loss_fn = self._scalar_loss_fn()
        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        sizes = [int(np.prod(leaf.shape)) for leaf in leaves]
        total = sum(sizes)

        def _spread_fit(candidates: list[int], cap: int) -> list[int]:
            """First fit over a spread traversal of ``candidates``, up to ``cap``
            parameters — so a truncated support samples the whole network instead of
            whichever leaves happen to sort first."""
            picked, used = [], 0
            for position in _spread_indices(len(candidates)):
                index = candidates[position]
                if used + sizes[index] <= cap:
                    picked.append(index)
                    used += sizes[index]
            return sorted(picked)

        def _select_support(scale: float = 1.0) -> list[int]:
            budget = int(self._fd_probe_capacity() * scale)
            span = self._fd_leaf_span(leaves, treedef, loss_fn)
            if span is None:
                # Untraceable loss, so nothing is known about who reads what: assume
                # the whole tree converts and spend what is left on a spread subset.
                cap = max(
                    (budget - total * self._FD_BYTES_PER_CONVERTED_PARAM)
                    // self._FD_BYTES_PER_SUPPORT_PARAM,
                    0,
                )
                picked = _spread_fit(list(range(len(leaves))), cap)
                return picked or [min(range(len(leaves)), key=lambda i: sizes[i])]

            first, last = span
            live = [index for index in range(len(leaves)) if last[index] >= 0]
            if not live:  # the traced loss reads no parameter at all
                return list(range(len(leaves)))

            # converted(cut): the parameters reverse mode upcasts once the shallowest
            # perturbed leaf is the one first read at equation ``cut`` — that is every
            # leaf still being read at or after that point. Prefix sums over a
            # last-read-descending order make it a bisection.
            by_last = sorted(live, key=lambda index: -last[index])
            last_desc = [last[index] for index in by_last]
            cumulative = [0]
            for index in by_last:
                cumulative.append(cumulative[-1] + sizes[index])

            def converted(cut: int) -> int:
                low, high = 0, len(last_desc)
                while low < high:
                    middle = (low + high) // 2
                    if last_desc[middle] >= cut:
                        low = middle + 1
                    else:
                        high = middle
                return cumulative[low]

            # Walk deepest first, taking every leaf whose addition keeps the whole
            # probe inside the budget. A leaf too big to perturb is skipped rather
            # than ending the walk: a shallower, smaller one may still fit, and the
            # skipped leaf is billed as a conversion at that shallower cut anyway.
            picked, support = [], 0
            for index in sorted(live, key=lambda i: (-first[i], -i)):
                trial = support + sizes[index]
                cost = (
                    trial * self._FD_BYTES_PER_SUPPORT_PARAM
                    + converted(first[index]) * self._FD_BYTES_PER_CONVERTED_PARAM
                )
                if cost <= budget:
                    picked.append(index)
                    support = trial
            if picked:
                return sorted(picked)
            # Not even the deepest single leaf fits. Price a spread subset against the
            # whole traced region, and if that is empty too, probe the smallest read
            # leaf and let ``support_coverage`` report how little was covered.
            cap = max(
                (budget - converted(0) * self._FD_BYTES_PER_CONVERTED_PARAM)
                // self._FD_BYTES_PER_SUPPORT_PARAM,
                0,
            )
            return _spread_fit(live, cap) or [min(live, key=lambda i: sizes[i])]

        def _run(scale: float = 1.0):
            support = _select_support(scale)
            if len(support) < 2:
                # One leaf cannot be split into two disjoint directions: the odd half
                # would be empty, ||g|| would clamp to the 1e-12 floor, both evaluations
                # would return the same loss, and the probe would "pass" on 0 == 0.
                # Refuse loudly instead — a direction with no parameters in it is not
                # evidence about anything, and `supported: False` is how the probe tells
                # test_gradient_authenticity that the machine, not the candidate, is at
                # fault. Unreachable on the reference hardware; a guard, not a path.
                raise RuntimeError(
                    f"finite-difference probe fits only {len(support)} parameter leaf/leaves "
                    f"of {len(leaves)} within its memory budget — too few to form two "
                    "independent directions"
                )

            def _loss_on(sub: list[Any]) -> Any:
                merged = list(leaves)
                for position, index in enumerate(support):
                    merged[index] = sub[position]
                return loss_fn(jax.tree_util.tree_unflatten(treedef, merged))

            base = [leaves[index].astype(jnp.float64) for index in support]
            _, grads = jax.value_and_grad(_loss_on)(base)

            def _probe(mask: list[bool]) -> dict[str, Any]:
                # Central difference: (L(+eps d) - L(-eps d)) / (2 eps) cancels the
                # even-order curvature term. Reported as base=L(-eps d), perturbed=
                # L(+eps d), epsilon=2 eps_eff so finite_difference_metric divides
                # correctly.
                #
                # Adaptive step: scale eps so the loss-step ``eps*analytic`` stays small
                # even when the directional derivative (== ``analytic`` == ||g_S||) is
                # large. A FIXED eps is too coarse for steep losses (large ||g|| vs a
                # tiny loss): the O(eps^2) curvature term of the central difference then
                # dominates and the estimate overshoots the true derivative. Cap at
                # the base eps, floor for float64 safety. This only makes the finite
                # difference MORE accurate for a genuine gradient — a faked/wrong
                # gradient's analytic still fails to match the (accurate) difference,
                # so the authenticity discriminator is preserved.
                #
                # The CAP is what binds on flat losses, where the adaptive term never
                # triggers and so cannot help. There is room to be that strict
                # because the step only has to lift the loss clear of float64 roundoff
                # (2*eps*analytic against ~1e-16 of the loss — a margin of 1e10 even
                # at the floor), while curvature error falls as eps^2. Tightening the
                # adaptive term to match would be a pessimisation, though: it drives
                # steep losses into the floor and back up the roundoff side of the
                # curve.
                square = sum(jnp.sum(g ** 2) for g, keep in zip(grads, mask) if keep)
                analytic = float(jnp.maximum(jnp.sqrt(square), 1e-12))
                eps_eff = min(float(epsilon), 3.0e-3 / max(analytic, 1e-9))
                eps_eff = max(eps_eff, 1e-6)
                # d = g_S/||g_S|| folded into the step, so the unit direction is never
                # a tree of its own, and +eps/-eps are evaluated one at a time: the
                # probe holds base + grads + ONE perturbed copy, never five trees.
                step = eps_eff / analytic

                def _loss_at(sign: float) -> float:
                    scale = jnp.asarray(sign * step, jnp.float64)
                    perturbed = [b + scale * g if keep else b for b, g, keep in zip(base, grads, mask)]
                    return float(np.asarray(_loss_on(perturbed)))

                return {
                    "loss_base": _loss_at(-1.0),
                    "loss_perturbed": _loss_at(+1.0),
                    "analytic_directional": analytic,
                    "epsilon": float(2.0 * eps_eff),
                    # Which leaves this direction actually spanned. A direction can come
                    # back on the 1e-12 floor for a perfectly good reason -- PPO's
                    # scalar loss reads `values` and not `logits`, so the lm_head leaf
                    # has an exactly zero gradient -- and the first question then is
                    # "how much was in that direction", which is unanswerable from a
                    # bare ||g||.
                    "direction_leaves": int(sum(1 for keep in mask if keep)),
                    "direction_params": int(
                        sum(sizes[index] for index, keep in zip(support, mask) if keep)
                    ),
                }

            probes = [_probe([True] * len(support))]
            # probes 1,2: the gradient restricted to an even/odd position within the
            # support (disjoint, together covering all of it).
            for parity in (0, 1):
                probes.append(_probe([position % 2 == parity for position in range(len(support))]))
            return probes, sum(sizes[index] for index in support)

        # Contained: the probe is the only artifact whose cost is set by the free
        # memory of the grading machine rather than by the candidate, so a failure
        # here must fail test_gradient_authenticity (``supported: False`` asserts)
        # and not take the other twenty tests down with it.
        #
        # ...and when the allocator disagrees with _fd_probe_capacity's per-parameter
        # estimate, believe the allocator. The estimate cannot price XLA's own float64
        # temporaries, so on the largest instances it can pick a support that then
        # refuses to allocate. Halve and retry rather than report a candidate as
        # ungradable over a budgeting miss on the grading machine; the smaller support
        # is honestly reported through support_coverage.
        probes = covered = None
        reason = "finite-difference probe did not run"
        for scale in (1.0, 0.5, 0.25):
            try:
                if self.jax_device is None:
                    probes, covered = _run(scale)
                else:
                    with jax.default_device(self.jax_device):
                        probes, covered = _run(scale)
                break
            except Exception as exc:  # noqa: BLE001
                # Keep the TEXT of the failure, never the exception object: its
                # __traceback__ pins every frame of the attempt that just failed, and
                # those frames pin its float64 trees. Retry with the exception still
                # bound and the next attempt runs with strictly less memory than the one
                # that already ran out -- so it fails too, and the halving is theatre.
                reason = f"{type(exc).__name__}: {exc}"
                retry = "RESOURCE_EXHAUSTED" in str(exc)
            if not retry:
                break
            gc.collect()
        if probes is None:
            return {"supported": False, "reason": reason}
        # Back-compat: expose probe 0's fields at the top level for the old shape.
        return {
            "supported": True,
            "probes": probes,
            "support_params": int(covered),
            "total_params": int(total),
            "support_coverage": (float(covered) / float(total)) if total else 0.0,
            # probes[0] spans the whole support, so its leaf count is the support's.
            "support_leaves": int(probes[0]["direction_leaves"]),
            "total_leaves": int(len(sizes)),
            **probes[0],
        }

    # --- relocated training probes --------------------------------------- #
    # The fixed-batch training probes below are reconstructed HERE, grader-side, from the
    # DOCUMENTED forward + config contract ALONE -- never from a trainer object that lives
    # only in the candidate repo. The reference JAX port is told to "mirror torch_orig's
    # structure and public API" with naming its own, so a faithful port need not expose a
    # ``dpo.trainer.DPOTrainer`` / ``sft.trainer.Trainer`` / ``ppo.ppo_utils`` seam at all;
    # reaching into those would grade the reference layout, not the port. Instead the loop is
    # rebuilt from parts every port must have: the candidate ``module.apply`` forward (via
    # ``_scalar_loss_fn_for`` / ``_dpo_trainer_loss_fn_for``), the resolved ``training_args``
    # hyperparameters, the vendored ``_formulas`` math, and ``optax`` (a standard library, not
    # a candidate module). The optimizer is a plain ``optax.adamw`` whose schedule is sampled
    # from ``F.lr_schedule`` -- the same oracle ``lr_schedule_artifact`` grades the port's own
    # schedule against -- so the AdamW update sequence is numerically identical to the loop the
    # reference used to ship, and every frozen train-loop golden (train_trajectory,
    # optimizer_probe, dpo_objective) is unchanged.
    @staticmethod
    def _decay_mask_flat(params):
        """optax weight-decay mask: ``True`` on every leaf EXCEPT biases / normalisation
        scales, keyed by the leaf's flat path. Mirrors the reference decay grouping
        (``no_decay = bias / layer_norm / layernorm / ln_ / scale``) without importing it.
        Works on both the nested full-finetuning tree and the flat ``{name: leaf}`` dict the
        LoRA path optimises."""
        import jax

        no_decay = ("bias", "layer_norm", "layernorm", "ln_", "scale")

        def _path_name(path) -> str:
            parts: list[str] = []
            for entry in path:
                key = getattr(entry, "key", None)
                if key is None:
                    key = getattr(entry, "name", None)
                if key is None:
                    key = getattr(entry, "idx", None)
                parts.append(str(key if key is not None else entry))
            return "/".join(parts).lower()

        return jax.tree_util.tree_map_with_path(
            lambda path, _leaf: not any(nd in _path_name(path) for nd in no_decay),
            params,
        )

    def _recon_optimizer(self, training_args, rates):
        """The reference optimizer, rebuilt contract-only: ``optax.adamw`` (betas / eps /
        weight-decay read from ``training_args``, decay mask from :func:`_decay_mask_flat`),
        with an LR schedule that replays ``rates`` -- the ``F.lr_schedule`` samples -- indexed
        by the optimizer step count, then ``optax.clip_by_global_norm`` when ``max_grad_norm``
        is set. Because ``rates[i]`` equals the port's own schedule at step ``i`` (the property
        ``lr_schedule_artifact`` grades), the update sequence matches the reference's."""
        import jax.numpy as jnp
        import optax

        rate_arr = jnp.asarray(rates, dtype=jnp.float32)
        n_rates = int(rate_arr.shape[0])

        def schedule(count):
            return rate_arr[jnp.clip(count, 0, n_rates - 1)]

        adam = optax.adamw(
            learning_rate=schedule,
            b1=float(getattr(training_args, "adam_beta1", 0.9)),
            b2=float(getattr(training_args, "adam_beta2", 0.999)),
            eps=float(getattr(training_args, "adam_epsilon", 1e-8)),
            weight_decay=float(getattr(training_args, "weight_decay", 0.0) or 0.0),
            mask=self._decay_mask_flat,
        )
        max_grad_norm = float(getattr(training_args, "max_grad_norm", 1.0) or 0.0)
        if max_grad_norm > 0:
            return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)
        return adam

    def _recon_run_training(
        self, module, params, batch, training_args, num_steps, ignore_index, **extra
    ) -> dict[str, Any]:
        """Fixed-batch training probe, reconstructed grader-side from the documented contract;
        returns ``{"losses": [...], "final_params": ...}`` (the shape ``train_trajectory`` /
        ``optimizer_probe`` / ``dpo_objective`` consume).

        The step objective is the candidate's own forward composed with the vendored loss:
        DPO uses the FULL trainer loss (:func:`_dpo_trainer_loss_fn_for` -- length-normalised
        NLL + odds-ratio/sigmoid + ``ftx``, the objective the ``train_trajectory`` golden was
        frozen from), SFT/PPO use :func:`_scalar_loss_fn_for` (the causal-LM loss / the inline
        GAE surrogate ``loss_trajectory`` already validates). LoRA descends only the adapter
        (+ PPO value-head) leaves ``_trainable_keys`` reports, closing the frozen base over as a
        constant, so the base is provably untouched (``base_max_delta == 0``). ``optax.adamw``
        + ``F.lr_schedule`` reproduce the reference update; fp32 upcast + ``highest`` matmul make
        the deterministic golden reproducible bit-for-bit."""
        import jax
        import jax.numpy as jnp
        import optax

        num_steps = int(num_steps)
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)

        if self.training_method == "dpo":
            loss_fn = self._dpo_trainer_loss_fn_for(batch, extra.get("ref_params"))
        else:
            loss_fn = self._scalar_loss_fn_for(batch)

        keep = self._trainable_keys(params)
        rates = F.lr_schedule(training_args, max(num_steps, 1))[: max(num_steps, 1)]
        optimizer = self._recon_optimizer(training_args, rates)

        losses: list[float] = []
        with jax.default_matmul_precision("highest"):
            if keep is None:
                opt_state = optimizer.init(params)
                for _ in range(num_steps):
                    loss_value, grads, _raw = self._value_and_grad_flat(loss_fn, params, None)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    losses.append(float(np.asarray(loss_value)))
            else:
                flat = F.flatten_tree_raw(params)
                trainable = {name: flat[name] for name in keep}
                opt_state = optimizer.init(trainable)
                for _ in range(num_steps):
                    loss_value, _tree, raw = self._value_and_grad_flat(loss_fn, params, keep)
                    grads = {name: raw[name] for name in keep}
                    updates, opt_state = optimizer.update(grads, opt_state, trainable)
                    trainable = optax.apply_updates(trainable, updates)
                    merged = F.flatten_tree_raw(params)
                    merged.update(trainable)
                    params = _unflatten_raw(merged)
                    losses.append(float(np.asarray(loss_value)))
        return {"losses": losses, "final_params": params}

    # --- lr schedule / replay -------------------------------------------- #
    def lr_schedule_artifact(self, num_steps: int = 8) -> dict[str, Any]:
        """Sample the LR schedule the port's resolved config implies.

        The reference LR schedule is fully determined by the DOCUMENTED config contract --
        ``lr_scheduler_type``, ``learning_rate``, ``warmup_steps``/``warmup_ratio`` and the
        step count -- so it is rebuilt grader-side from ``self.training_args`` via the vendored
        ``F.lr_schedule`` (the same formula ``lr_schedule.json`` was frozen from) rather than by
        reaching into a candidate-only trainer/scheduler object the port is free not to expose.
        A port that resolves the same hyperparameters through ``config.get_train_args`` yields
        the same curve by construction; a port that resolves them WRONG produces a different
        ``training_args`` and fails here (and again in the end-to-end training goldens).
        ``num_steps`` defaults to 8 to match the sampling ``lr_schedule.json`` was written
        with."""
        steps = int(num_steps)
        try:
            rates = [float(r) for r in F.lr_schedule(self.training_args, max(steps, 1))[:steps]]
        except Exception as exc:  # noqa: BLE001
            return {"supported": False, "reason": f"schedule reconstruction failed: {exc}"}
        return {"supported": True, "learning_rates": rates}

    def loss_trajectory_artifact(self) -> dict[str, Any]:
        import jax
        import math as _math

        num_steps = self.loss_trajectory_steps
        schedule = F.lr_schedule(self.training_args, max(num_steps, 1))[:num_steps]
        loss_fn = self._scalar_loss_fn()
        params = self.params
        # LoRA: descend only the adapters, exactly as the reference trainer does -- a
        # replay that also walks the frozen base is a different optimization problem
        # from the one the golden curve was frozen from (see _trainable_keys).
        keep = self._trainable_keys(params)
        trajectory: list[dict[str, Any]] = []
        for step_index, learning_rate in enumerate(schedule):
            if self.jax_device is None:
                loss_value, grads, raw = self._value_and_grad_flat(loss_fn, params, keep)
            else:
                with jax.default_device(self.jax_device):
                    loss_value, grads, raw = self._value_and_grad_flat(loss_fn, params, keep)
            flat = {k: _to_np(v) for k, v in raw.items()}
            grad_norm_sq = sum(float(np.square(v.astype(np.float32)).sum()) for v in flat.values())
            lr = float(learning_rate)
            if keep is None:
                params = jax.tree_util.tree_map(lambda p, g: p - lr * g.astype(p.dtype), params, grads)
            else:
                updated = F.flatten_tree_raw(params)
                for name, grad in raw.items():
                    leaf = updated[name]
                    updated[name] = leaf - lr * grad.astype(leaf.dtype)
                params = _unflatten_raw(updated)
            trajectory.append(
                {
                    "step": step_index,
                    "learning_rate": lr,
                    "loss": float(np.asarray(loss_value)),
                    "grad_norm": float(_math.sqrt(grad_norm_sq)),
                }
            )
        return {"supported": True, "trajectory": trajectory}

    def determinism_probe(self) -> dict[str, Any]:
        try:
            first = self.forward_outputs(output_hidden_states=False)["logits"]
            second = self.forward_outputs(output_hidden_states=False)["logits"]
        except Exception as exc:
            return {"supported": False, "reason": f"forward failed: {exc}"}
        first = np.asarray(first).astype(np.float32, copy=False)
        second = np.asarray(second).astype(np.float32, copy=False)
        if first.shape != second.shape:
            return {"supported": True, "identical": False, "reason": "shape changed between forwards"}
        max_abs_diff = float(np.max(np.abs(first - second))) if first.size else 0.0
        return {"supported": True, "identical": max_abs_diff == 0.0, "max_abs_diff": max_abs_diff}

    def config_spec_probe(self) -> dict[str, Any]:
        """Resolved config-DEFAULT fields (test_config_spec, config-DEFAULT parity).

        The port's ``config.get_train_args({})`` -- the pure ``config.py`` dataclass DEFAULTS
        under an EMPTY override dict, NOT the explicit ``meta.json`` repo_config the numeric
        probes resolve in ``__init__`` -- must resolve the SAME load-bearing defaults the
        reference declares (learning rate, scheduler, warmup, batch/accum, finetuning + LoRA
        knobs, preference/PPO/generation knobs, template, cutoff). ``{}`` (not ``None``:
        ``read_args(None)`` routes to ``sys.argv`` on both stacks) short-circuits I/O-free to
        the defaults plus the in-config post-processing. Read from whichever returned dataclass
        owns each field, first owner wins -- no model, no forward, torch-free. Fully guarded:
        any failure reports ``supported: False`` rather than breaking the one-shot ``run_all``
        worker.
        """
        try:
            resolved = self._config_mod.get_train_args({})
            sources = [s for s in resolved if s is not None]
            config: dict[str, Any] = {}
            for field in _CONFIG_SPEC_FIELDS:
                for src in sources:
                    if hasattr(src, field):
                        config[field] = _config_json_safe(getattr(src, field))
                        break
            return {"supported": True, "config": config}
        except Exception as exc:  # noqa: BLE001 - contained: never break run_all
            return {"supported": False, "reason": f"config extraction failed: {exc}"}

    def loader_spec_probe(self) -> dict[str, Any]:
        """Tokenizer facts the port's ``load_tokenizer`` must reproduce (test_loader_spec).

        A port that loads the wrong tokenizer -- or drops the chat template, or resolves a
        different pad/eos/bos id -- tokenizes and masks differently, which every downstream
        numeric check then inherits. This pins the observable tokenizer contract from the
        tokenizer this runtime already loaded. Guarded like ``config_spec_probe``.
        """
        try:
            tok = self.tokenizer
            spec = {
                "vocab_size": _token_id_safe(getattr(tok, "vocab_size", None)),
                "len": int(len(tok)) if hasattr(tok, "__len__") else None,
                "pad_token_id": _token_id_safe(getattr(tok, "pad_token_id", None)),
                "eos_token_id": _token_id_safe(getattr(tok, "eos_token_id", None)),
                "bos_token_id": _token_id_safe(getattr(tok, "bos_token_id", None)),
                "unk_token_id": _token_id_safe(getattr(tok, "unk_token_id", None)),
                "padding_side": getattr(tok, "padding_side", None),
                "has_chat_template": bool(getattr(tok, "chat_template", None)),
            }
            return {"supported": True, "spec": spec}
        except Exception as exc:  # noqa: BLE001 - contained: never break run_all
            return {"supported": False, "reason": f"loader spec extraction failed: {exc}"}

    def generation_artifact(self) -> dict[str, Any]:
        import inspect

        if not hasattr(self.model, "generate"):
            return {"supported": False, "reason": "model has no generate"}
        inputs = {k: self.batch[k] for k in _GENERATION_KEYS if k in self.batch}
        sig = inspect.signature(self.model.generate)
        try:
            sequences = self.model.generate(**inputs, max_new_tokens=self.generation_max_new_tokens)
        except Exception as exc:
            return {"supported": False, "reason": f"generate failed: {exc}"}
        return {"supported": True, "sequences": np.asarray(_to_np(sequences)).tolist()}

    def generation_logits_probe(self) -> dict[str, Any]:
        """Teacher-forced continuation-logit parity.

        The reference greedy ``generate`` is an uncached full re-forward per step
        (``for _ in range(max_new_tokens): logits = apply(full_ids); argmax``), so
        forwarding the model ONCE over the reference's frozen generated sequence and
        reading the logits at continuation position ``prompt_len+k-1`` reproduces step
        k's next-token prediction EXACTLY: strict causal attention means that position's
        logits depend only on tokens ``<= prompt_len+k-1`` — exactly the sequence the
        reference held at step k. This teacher-forcing keeps a single bf16 argmax tie
        from cascading (each step is scored against the reference's own prefix).

        For each step it returns the candidate's argmax, the reference's actual next
        token, and the softmax probability gap between them. Tie-robust by construction:
        a genuine bf16 argmax tie (degenerate newline/space tail) has a tiny prob gap
        while a real forward divergence drives it toward 1. Text-only — multimodal
        generation would need the vision inputs threaded through the grown sequence, so
        it is reported unsupported and deselected (mirrors the self-consistency
        invariants).

        The teacher-forcing mask is the prompt's OWN ``attention_mask`` extended with
        ones for the generated tail — NOT all-ones. The reference generate carries the
        prompt's (possibly right-padded) mask forward, so a prompt PAD token must stay
        masked; an all-ones mask would let the continuation attend to those pads and flip
        it (gemma3's golden prompt is right-padded 20+4 → step 0 diverges with prob_gap
        ~1.0 under all-ones, matches exactly under the padded mask). No-op for the
        unpadded prompts (all-ones already).
        """
        seq = self.generation_sequences
        if seq is None:
            return {"supported": False, "reason": "golden froze no generation sequences"}
        if any(k in self._np_batch for k in self._MULTIMODAL_KEYS):
            return {"supported": False, "reason": "multimodal generation: text-only probe N/A"}
        seq = np.asarray(seq)
        if seq.ndim == 1:
            seq = seq[None, :]
        try:
            prompt_len = int(self._np_batch["input_ids"].shape[1])
            n_new = int(self.generation_max_new_tokens)
            prompt_mask = self._np_batch.get("attention_mask")
            if (
                prompt_mask is not None
                and np.asarray(prompt_mask).ndim == 2
                and np.asarray(prompt_mask).shape == (seq.shape[0], prompt_len)
                and seq.shape[1] >= prompt_len
            ):
                tail = np.ones((seq.shape[0], seq.shape[1] - prompt_len), dtype=np.int64)
                mask = np.concatenate([np.asarray(prompt_mask).astype(np.int64), tail], axis=1)
            else:
                mask = np.ones(seq.shape, dtype=np.int64)
            logits = self.forward_on(seq, mask)  # (b, S, V) fp32
            seq_span = int(logits.shape[1])
            steps: list[dict[str, Any]] = []
            for k in range(n_new):
                pos = prompt_len + k - 1       # logits here predict token prompt_len+k
                tok_idx = prompt_len + k
                if pos < 0 or tok_idx >= seq.shape[1] or pos >= seq_span:
                    break
                step_logits = logits[:, pos, :].astype(np.float64)   # (b, V)
                target = seq[:, tok_idx].astype(np.int64)            # (b,)
                argmax = np.argmax(step_logits, axis=-1)
                shifted = step_logits - step_logits.max(axis=-1, keepdims=True)
                probs = np.exp(shifted)
                probs = probs / probs.sum(axis=-1, keepdims=True)
                rows = np.arange(step_logits.shape[0])
                steps.append({
                    "argmax": argmax.astype(np.int64).tolist(),
                    "target": target.tolist(),
                    "prob_gap": (probs[rows, argmax] - probs[rows, target]).astype(np.float64).tolist(),
                    "p_argmax": probs[rows, argmax].astype(np.float64).tolist(),
                })
            return {"supported": True, "prompt_len": prompt_len, "n_new": n_new, "steps": steps}
        except Exception as exc:  # any model that can't take the grown (ids, mask) -> skip
            return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}

    # --- fp32 exactness probe ---------------------------- #
    def fp32_forward_outputs(self) -> dict[str, Any]:
        """FP32 exactness probe: upcast params AND float inputs to float32 and run the
        candidate's OWN forward. Requires dtype-following modules (dtype=None on
        Dense/Embed) — the same path the training replay uses. The loss is the shifted
        causal CE in fp32 (F.jax_causal_loss), matching the reference's
        _torch_causal_loss (not the module's internal loss)."""
        import jax
        import jax.numpy as jnp

        params_f32 = self._fp32_params_cached()  # shared fp32 upcast (see _fp32_params_cached)

        def _up(v):
            arr = v
            return arr.astype(jnp.float32) if jnp.issubdtype(jnp.asarray(arr).dtype, jnp.floating) else arr

        inputs = {
            k: _up(v)
            for k, v in self.batch.items()
            if not (self.training_method == "ppo" and k == "labels")
        }

        def _forward():
            output = F.call_model_with_control_kwargs(
                self.model.apply, {"params": params_f32}, model_inputs=inputs,
                return_dict=True, use_cache=False, output_hidden_states=False,
            )
            logits = F.extract_output_field(output, "logits")
            if logits is None and isinstance(output, tuple) and output:
                logits = output[0]
            return logits

        if self.jax_device is None:
            logits = _forward()
        else:
            with jax.default_device(self.jax_device):
                logits = _forward()
        result: dict[str, Any] = {"logits": _to_np(logits)}
        # fp32 LOSS gate is the shifted causal CE (SFT only); DPO/PPO losses are
        # method-specific (already graded at bf16), so those get forward-logit parity.
        if self.training_method == "sft":
            labels = self.batch.get("labels", self.batch["input_ids"])
            loss = F.jax_causal_loss(logits, labels, self.ignore_index)
            result["loss"] = float(_to_np(loss).reshape(()))
        return result

    # --- torch-free self-consistency invariants ---------- #
    # Frozen-batch keys that make a batch multimodal -> text-only invariants N/A.
    _MULTIMODAL_KEYS = (
        "pixel_values", "pixel_values_videos", "image_grid_thw",
        "video_grid_thw", "image_sizes", "rope_deltas",
    )

    def forward_on(self, input_ids: Any, attention_mask: Any = None, *, fp32: bool = False) -> np.ndarray:
        """Forward ARBITRARY (constructed) token inputs through the loaded module+params.

        Torch-free and deterministic: reuses ``self.model``/``self.params`` on inputs
        derived from the frozen batch (valid in-vocab ids). Returns fp32 logits of
        shape ``(b, s, vocab)``.

        ``fp32=True`` upcasts the params to float32 first (the fp32_forward path), so the
        whole forward runs in float32 — the clean cross-framework parity surface the
        fp32 battery grades. Text inputs carry no float tensors, so only the params
        need upcasting.
        """
        params = self.params
        if fp32:
            # Shared, cached fp32 upcast (built once, reused across the 4 battery fp32
            # forwards + fp32_forward); bit-identical to a per-call astype(float32).
            params = self._fp32_params_cached()
        model_inputs: dict[str, Any] = {
            "input_ids": F.to_jax_on_device(np.asarray(input_ids), self.jax_device),
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = F.to_jax_on_device(
                np.asarray(attention_mask), self.jax_device
            )
        output = F.call_model_with_control_kwargs(
            self.model.apply,
            {"params": params},
            model_inputs=model_inputs,
            return_dict=True,
            use_cache=False,
            output_hidden_states=False,
        )
        logits = F.extract_output_field(output, "logits")
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
        # Always fp32 (as the docstring promises). _to_np only upcasts bf16; fp16
        # models (e.g. bloom) otherwise leak fp16 logits, and the fp32 self-consistency
        # invariants (softmax normalization especially: exp/sum over a large vocab in
        # fp16 accumulates ~0.13 error) would spuriously fail on a correct model.
        return np.asarray(_to_np(logits), dtype=np.float32)

    def self_consistency_probe(self) -> dict[str, Any]:
        """Golden-free framework invariants on constructed inputs.

        Exact bitwise checks (causal, masked-padding-content) are reduced to scalars
        here; distribution checks (right-pad vs unpadded, batch vs single) ship raw
        fp32 logits so the test can reuse the calibrated ``compare.check_logits``
        (incl. the extreme-logit KL-cosine override) against ``golden.thresholds``.
        """
        if any(k in self._np_batch for k in self._MULTIMODAL_KEYS):
            return {"supported": False, "reason": "multimodal batch: text-only invariants N/A"}
        ids_all = np.asarray(self._np_batch["input_ids"])
        if ids_all.ndim != 2 or ids_all.shape[1] < 2:
            return {"supported": False, "reason": "need a 2-D batch with seq_len >= 2"}
        try:
            row = ids_all[:1].astype(np.int64, copy=True)          # (1, L)
            L = int(row.shape[1])
            ones = np.ones((1, L), dtype=row.dtype)
            base = self.forward_on(row, ones)                      # (1, L, V) fp32
            vocab = int(base.shape[-1])

            def _other_id(exclude: int) -> int:
                for x in row[0].tolist():                          # prefer a real in-vocab id
                    if int(x) != int(exclude):
                        return int(x)
                return int((int(exclude) + 1) % max(vocab, 2))

            # (a) causal masking: perturb the token at position t.
            t = max(1, L // 2)
            pert = row.copy()
            pert[0, t] = _other_id(int(row[0, t]))
            base_pert = self.forward_on(pert, ones)
            causal_before = float(np.abs(base[:, :t, :] - base_pert[:, :t, :]).max())
            causal_after = float(np.abs(base[:, t:, :] - base_pert[:, t:, :]).max())

            # (b) padding invariance.
            P = 4
            pad_mask = np.concatenate(
                [np.ones((1, L), row.dtype), np.zeros((1, P), row.dtype)], axis=1
            )
            pad_a = np.concatenate([row, np.zeros((1, P), row.dtype)], axis=1)
            pad_b = np.concatenate([row, np.full((1, P), _other_id(0), row.dtype)], axis=1)
            logits_pad_a = self.forward_on(pad_a, pad_mask)        # (1, L+P, V)
            logits_pad_b = self.forward_on(pad_b, pad_mask)
            pad_samelen = float(np.abs(logits_pad_a[:, :L, :] - logits_pad_b[:, :L, :]).max())

            # (c) softmax normalization on the unpadded forward logits.
            lse = F.np_logsumexp(base, axis=-1, keepdims=True)
            probs_sum = np.exp(base - lse).sum(axis=-1)
            softmax_dev = float(np.abs(probs_sum - 1.0).max())

            # (d) batch independence: stack row + a distinct row2.
            p2 = min(3, L - 1)
            row2 = row.copy()
            row2[0, p2] = _other_id(int(row[0, p2]))
            single2 = self.forward_on(row2, ones)
            batch_out = self.forward_on(
                np.concatenate([row, row2], axis=0), np.ones((2, L), dtype=row.dtype)
            )

            config = getattr(self.model, "config", None)
            return {
                "supported": True,
                # config.causal is False only for legitimately bidirectional prefix-LM
                # references (paligemma); every strictly-causal model omits the attr.
                "causal_expected": bool(getattr(config, "causal", True)),
                "seq_len": L, "perturb_pos": t, "pad_len": P,
                "causal_before_max_abs_diff": causal_before,
                "causal_after_max_abs_diff": causal_after,
                "pad_samelen_max_abs_diff": pad_samelen,
                "softmax_max_abs_dev": softmax_dev,
                "pad_unpadded_ref": base,                          # (1, L, V)
                "pad_difflen_unpadded": logits_pad_a[:, :L, :],    # (1, L, V)
                "single1": base,                                   # == forward_on(row)
                "single2": single2,                                # (1, L, V)
                "batch_row0": batch_out[:1],                       # (1, L, V)
                "batch_row1": batch_out[1:2],                      # (1, L, V)
            }
        except Exception as exc:  # any model that can't take bare (ids, mask) -> skip
            return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}

    def battery_probe(self) -> dict[str, Any]:
        """Input-space generality battery.

        Re-runs the candidate's forward-logit parity over the shape/length/batch/mask
        variants the reference froze (short_len1, long_maxctx, batch_multi,
        padded_ragged), catching modeling.py bugs a
        single fixed probe cannot — RoPE at a long position, a batch that collapses onto
        row 0, a ragged attention mask, the length-1 edge case.

        For each variant: bf16-native logits (params as loaded) and fp32 logits (params
        upcast), plus — for SFT, where a valid shifted-causal span exists — the
        shifted-CE loss at both precisions (F.jax_causal_loss, matching the reference's
        _causal_cross_entropy). Text-only: a multimodal batch would desynchronise the
        fixed image/grid token block from the vision tower, so it self-reports
        unsupported (and its tests deselect on the golden anyway).
        """
        if self.battery_inputs is None:
            return {"supported": False, "reason": "no input battery frozen (multimodal / older golden)"}
        if any(k in self._np_batch for k in self._MULTIMODAL_KEYS):
            return {"supported": False, "reason": "multimodal batch: input battery is text-only"}
        want_loss = self.training_method == "sft"
        variants: dict[str, Any] = {}
        for name, inp in self.battery_inputs.items():
            try:
                ids = np.asarray(inp["input_ids"])
                am = inp.get("attention_mask")
                am = None if am is None else np.asarray(am)
                bf16_logits = self.forward_on(ids, am, fp32=False)
                fp32_logits = self.forward_on(ids, am, fp32=True)
                entry: dict[str, Any] = {"logits": bf16_logits, "fp32_logits": fp32_logits}
                labels = inp.get("labels")
                if (
                    want_loss and labels is not None
                    and np.asarray(labels).ndim == 2 and np.asarray(labels).shape[1] >= 2
                    and bool((np.asarray(labels)[:, 1:] != self.ignore_index).any())
                ):
                    labels = np.asarray(labels)
                    entry["loss"] = float(
                        _to_np(F.jax_causal_loss(bf16_logits, labels, self.ignore_index)).reshape(())
                    )
                    entry["fp32_loss"] = float(
                        _to_np(F.jax_causal_loss(fp32_logits, labels, self.ignore_index)).reshape(())
                    )
                variants[name] = entry
            except Exception as exc:  # a model that cannot take a given shape -> that variant fails
                variants[name] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                gc.collect()
        return {"supported": True, "variants": variants}

    # --- vision-tower sensitivity ------------------------ #
    #: Frozen-batch keys whose CONTENT the graded loss must depend on: the float
    #: image/video pixels only -- NOT the grid/size/rope integer bookkeeping, whose
    #: perturbation would desync the image-token count from the tower (a crash, not a
    #: measurement) instead of exercising the tower itself.
    _VISION_CONTENT_KEYS = ("pixel_values", "pixel_values_videos")

    def mm_tower_sensitivity_probe(self) -> dict[str, Any]:
        """Golden-free vision-tower sensitivity gate.

        The graded numeric/behavioral objective is a causal-LM loss over label tokens;
        on a multimodal batch those label logits are only image-aware if the vision
        tower's embeddings actually reach them -- i.e. the port ran the tower AND the
        image tokens attend to the labels. This probe proves that dependence empirically
        and WITHOUT any reference value: it recomputes the exact scalar objective the
        numeric goldens were frozen from (:func:`_scalar_loss_fn_for`) on the real batch,
        then again on a batch whose float image content (``pixel_values`` /
        ``pixel_values_videos``) is perturbed by full-signal, per-element deterministic
        Gaussian noise. Per-element noise changes the spatial PATTERN, which layer/patch
        normalization inside the tower propagates rather than cancels (a constant shift
        would be absorbed), so a faithful port's loss moves measurably (``delta`` >> 0);
        a port that ignores the pixels -- constant/stub tower, dropped vision path, or
        image tokens that never attend -- leaves the whole forward graph independent of
        the pixels and returns a bitwise-identical loss (``delta`` == 0.0). The test
        turns ``delta`` into a {0,1} discriminator against a small floor. Two forwards,
        no gradient, no golden; skipped when the batch carries no float image content.
        """
        import jax
        import jax.numpy as jnp

        keys = [
            k
            for k in self._VISION_CONTENT_KEYS
            if k in self.batch and jnp.issubdtype(jnp.asarray(self.batch[k]).dtype, jnp.floating)
        ]
        if not keys:
            return {"supported": False, "reason": "no float image content in batch"}
        try:
            base = float(_to_np(self._scalar_loss_fn_for(self.batch)(self.params)).reshape(()))
            perturbed = dict(self.batch)
            rng = jax.random.PRNGKey(0)
            for i, k in enumerate(keys):
                orig = jnp.asarray(self.batch[k])
                arr = orig.astype(jnp.float32)
                # Scale the noise to the input's own spread so it is unmistakable yet
                # bounded (no bf16 overflow on the tower's forward); a distinct fold per
                # key keeps multi-input batches deterministic without correlating draws.
                scale = jnp.maximum(jnp.std(arr), jnp.asarray(1e-3, jnp.float32))
                noise = jax.random.normal(jax.random.fold_in(rng, i), arr.shape, dtype=jnp.float32) * scale
                perturbed[k] = (arr + noise).astype(orig.dtype)
            pert = float(_to_np(self._scalar_loss_fn_for(perturbed)(self.params)).reshape(()))
        except Exception as exc:
            return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            gc.collect()
        delta = abs(pert - base)
        return {
            "supported": True,
            "loss_base": base,
            "loss_perturbed": pert,
            "delta": float(delta),
            "perturbed_keys": keys,
        }

    # --- driver ----------------------------------------------------------- #
    def run_all(self) -> dict[str, Any]:
        results: dict[str, Any] = {}

        def produce(name: str, artifact: Any) -> None:
            """Run one artifact, keep its result, and then collect.

            The collection is not hygiene, it is load-bearing. A device buffer is
            freed when the last Python reference to its array dies, and several of
            these artifacts leave that last reference inside a reference CYCLE that
            only the collector can break: every ``except Exception`` here binds a
            traceback, a traceback pins its frames, and a frame pins the trees that
            frame allocated. Nothing between artifacts breaks the cycle, so the peak
            of one artifact becomes the floor of the next.

            After ``train_trajectory`` -- which upcasts the whole tree to float32 and
            runs optax AdamW over it -- the device can sit at its memory limit; one
            ``gc.collect()`` returns it to the resident-params floor. Without it the
            artifacts that follow run with nothing left, and ``fp32_forward`` fails on
            the embedding upcast -- which fails every test of the task, because run_all
            is one shot (see _CandidateRuntime).

            Milliseconds against artifacts that take minutes, so it runs after every
            one rather than after the few that are heavy today.
            """
            results[name] = artifact()
            gc.collect()

        produce("param_spec", self.param_spec)
        produce("param_norms", self.param_norms)
        produce("structural", self.structural_probe)
        # Spec probes: resolved training config + tokenizer facts. Cheap (no forward),
        # self-guarded, and presence-gated on the grade side -- deselected until a future
        # freezer emits their goldens, so producing them here is a harmless no-cost-of-safety.
        produce("config_spec", self.config_spec_probe)
        produce("loader_spec", self.loader_spec_probe)
        produce("batch_schema", self.candidate_batch_schema)
        produce("batch_arrays", self.candidate_batch_arrays)  # content (reuses the cached build)
        produce("forward_purity", self.forward_purity_probe)
        produce("forward", self.forward_outputs)
        produce("loss_artifact", self.loss_artifact)
        produce("gradient", self.gradient_artifact)
        produce("lr_schedule", self.lr_schedule_artifact)
        produce("loss_trajectory", self.loss_trajectory_artifact)
        # train_trajectory (the repo's OWN training loop, reconstructed) is deliberately NOT
        # produced here. It runs in its own spawned worker under a hard wall-clock timeout
        # (Candidate._ensure_train_run) so a port that never finishes can
        # be killed outright -- failing only test_train_loss_trend instead of hanging every
        # artifact in this shared run. dpo_objective still calls run_training (one step) here.
        produce("dpo_objective", self.dpo_objective_probe)
        produce("determinism", self.determinism_probe)
        produce("self_consistency", self.self_consistency_probe)
        # Vision-tower sensitivity gate (golden-free {0,1}); multimodal batches only
        # (the probe self-reports unsupported and is skipped on a text batch anyway). Runs
        # here, before the heavy fp32 phase, so its two forwards have the most headroom.
        if any(k in self._np_batch for k in self._MULTIMODAL_KEYS):
            produce("mm_tower_sensitivity", self.mm_tower_sensitivity_probe)
        # Input-space generality battery; text-only inside the probe, so it
        # self-reports unsupported for a multimodal batch or a golden with no battery.
        produce("battery", self.battery_probe)
        produce("fp32_forward", self.fp32_forward_outputs)  # all methods
        # The fp32 phase (battery + fp32_forward) is done; drop the shared fp32 upcast so
        # the 2x-checkpoint tree does not become the memory floor for the trailing
        # artifacts (generation / gradient_authenticity, which spends all it is given).
        self._drop_fp32_params_cache()
        if self.training_method == "sft":
            produce("generation", self.generation_artifact)
            # Teacher-forced continuation-logit parity; text-only inside the
            # probe, so it self-reports unsupported for a multimodal SFT batch.
            produce("generation_logits", self.generation_logits_probe)
        # Last, deliberately. The finite-difference probe deliberately spends whatever
        # device memory is left (it sizes its own support against it), and the float64
        # trees it allocates and frees leave the BFC pool fragmented enough that the
        # NEXT artifact can fail to find a contiguous block -- observed on
        # dpo_llava_next, where a probe that succeeded on its second budget was followed
        # by a loss_trajectory that could not allocate 370 MiB. Running it after
        # everything else confines that cost to the one artifact that chose to pay it;
        # its own failure is already contained (``supported: False``).
        produce("gradient_authenticity", self.native_gradient_authenticity)
        return results

    def _dpo_run_extra(self) -> dict[str, Any]:
        """``run_training`` kwargs for a DPO instance: always ``dpo_args``; ``ref_params``
        ONLY for a reference-BASED objective.

        ORPO / SimPO — every current instance — are reference-free: neither the reference
        ``DPOTrainer`` nor the candidate loss ever reads ``ref_params`` for them. Handing
        ``run_training`` a full copy of the params as the "reference policy" is then pure
        dead weight, and once it is upcast to float32 alongside the trainable params it
        doubles the fp32 parameter footprint — enough to OOM the largest multimodal DPO
        models (e.g. dpo_llava_next). Gating on the loss type mirrors ``run_dpo``'s own
        ``use_ref_model = pref_loss not in {orpo, simpo}`` and is numerically identical
        for the reference-free losses (which ignore the argument entirely)."""
        dpo = self.extra_args[0] if self.extra_args else None
        extra: dict[str, Any] = {"dpo_args": dpo}
        loss_type = str(getattr(dpo, "pref_loss", "sigmoid")).lower()
        if loss_type not in ("orpo", "simpo"):
            extra["ref_params"] = self.ref_params if self.ref_params is not None else self.params
        return extra

    def _fp32_params_cached(self) -> Any:
        """Float32 upcast of ``self.params``, built once and reused across the run_all fp32
        probes -- the input-space battery's 4 fp32 forwards (``forward_on(fp32=True)``) and
        ``fp32_forward_outputs`` -- instead of re-upcasting the whole tree per call (5
        identical full-tree casts). ``astype(float32)`` is deterministic, so the shared
        tree is byte-identical to a fresh per-call upcast: this changes churn, not
        numerics. ``self.params`` stays resident (native), so the peak is native + one
        fp32 tree -- exactly the transient peak each per-call upcast already hit -- just
        held steady across the adjacent battery -> fp32_forward window rather than rebuilt
        and freed 5 times. ``run_all`` drops it via ``_drop_fp32_params_cache`` the moment
        the fp32 phase ends so the 2x-checkpoint tree never becomes the memory floor of the
        trailing artifacts (see ``produce`` / gradient_authenticity)."""
        import jax
        import jax.numpy as jnp

        if self._fp32_params_cache is None:
            self._fp32_params_cache = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.float32), self.params
            )
        return self._fp32_params_cache

    def _drop_fp32_params_cache(self) -> None:
        """Release the cached fp32 param tree (see ``_fp32_params_cached``) and collect, so
        the fp32 upcast does not stay resident alongside the native params for the artifacts
        that run after the fp32 phase."""
        self._fp32_params_cache = None
        gc.collect()

    def _fp32_params_freeing_native(self) -> tuple[Any, Any]:
        """Upcast ``self.params`` to float32 and free the native-precision device copy.

        ``_recon_run_training`` upcasts its params to float32 as its first act (a no-op on
        the already-fp32 tree returned here), so the training replay need not keep BOTH the
        native (bf16) params AND their fp32 upcast resident at once. That redundant native
        copy — ~14 GiB for a 7B model — is what tips the largest multimodal DPO instance
        (dpo_llava_next: llava-1.6-vicuna-7b + vision tower + a long chosen/rejected batch)
        past the memory limit while the plain 7B DPO instances (e.g. mistral) fit. Returns
        ``(fp32_params, native_dtypes)``; restore the exact native tree afterwards with
        ``_restore_native_params``. bf16->fp32->bf16 is lossless, so every artifact that
        runs after the replay sees byte-identical params — this changes peak memory, not
        numerics. Harmless (and cheap) for the small instances, so it stays uniform."""
        import jax
        import jax.numpy as jnp

        native_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, self.params)
        fp32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), self.params)
        self.params = None
        self._fp32_params_cache = None  # never let a run_all fp32 cache outlive its source params
        gc.collect()
        return fp32, native_dtypes

    def _restore_native_params(self, fp32_params: Any, native_dtypes: Any) -> None:
        """Inverse of ``_fp32_params_freeing_native``: rebuild ``self.params`` at its
        original per-leaf dtype from the (immutable, untrained) fp32 tree the replay used."""
        import jax

        self.params = jax.tree_util.tree_map(lambda a, d: a.astype(d), fp32_params, native_dtypes)
        self._fp32_params_cache = None  # self.params is a fresh tree; force a rebuild if fp32 is needed again
        gc.collect()

    def train_trajectory_artifact(self) -> dict[str, Any]:
        """Reconstruct the candidate's fixed-batch training loop (its genuine trainer
        components: optimizer + objective, via ``_recon_run_training``) over the fixed batch
        — the end-to-end training check. Skips cleanly if the port's trainer components are
        missing/broken (so component-only repos don't fail here)."""
        num_steps = self.loss_trajectory_steps
        # DPO reconstruction needs the DPO hyperparameters and, for a reference-based loss,
        # the frozen reference params so it can reproduce the reference DPOTrainer
        # objective. SFT ignores these. (See ``_dpo_run_extra`` for the ref_params gate.)
        # Build extra BEFORE freeing the native params: a reference-based loss (not any
        # current instance) folds self.params in as ref_params, which must be captured first.
        extra: dict[str, Any] = {}
        if self.training_method == "dpo":
            extra = self._dpo_run_extra()
        # Hand the reconstruction the fp32 upcast directly and drop the redundant native copy,
        # so a 7B multimodal DPO replay fits (see _fp32_params_freeing_native). The
        # reconstruction's own upcast is then a no-op, so the trajectory is numerically identical.
        run_params, native_dtypes = self._fp32_params_freeing_native()
        try:
            if self.jax_device is None:
                result = self._recon_run_training(self.model, run_params, self.batch, self.training_args, num_steps, self.ignore_index, **extra)
            else:
                import jax

                with jax.default_device(self.jax_device):
                    result = self._recon_run_training(self.model, run_params, self.batch, self.training_args, num_steps, self.ignore_index, **extra)
        except Exception as exc:  # noqa: BLE001
            return {"supported": False, "reason": f"run_training failed: {exc}"}
        finally:
            self._restore_native_params(run_params, native_dtypes)
        losses = [float(x) for x in (result or {}).get("losses", [])]
        return {"supported": True, "losses": losses}

    def _probe_lr0(self) -> float:
        """Step-0 learning rate the reconstructed optimizer applies on a 1-step run.

        Sampled from the vendored schedule (``F.lr_schedule(training_args, 1)[0]``) -- the exact
        construction ``_recon_optimizer`` feeds ``optax.adamw`` -- so ``median|dP| ~ lr0``
        compares like against like. All 45 goldens have no warmup, so ``lr0 = learning_rate > 0``
        and the discriminator is live."""
        rates = F.lr_schedule(self.training_args, 1)
        return float(rates[0]) if rates else 0.0

    def optimizer_probe_artifact(self) -> dict[str, Any]:
        """LoRA base-freeze + AdamW-vs-SGD optimizer-identity probe.

        Reconstructs the candidate's fixed-batch training step (``_recon_run_training``) for
        ONE step and reads the returned ``final_params``, grading two properties the loss-curve
        gates cannot see -- at the
        tiny golden LR the K-step curve is nearly flat, so a flat/SGD replay or a
        full-finetune-as-LoRA still lands inside ``test_train_trajectory``'s bf16 band:

          * base-freeze: every non-adapter (base) leaf must be byte-unchanged after the step
            (``base_max_delta`` <= 1e-12) and >=1 adapter leaf must have moved. Diffed
            leaf-by-leaf with a SCALAR reduction only -- never a third resident fp32 tree,
            since the 7B fp32 upcast already sits at the memory ceiling.
          * optimizer-identity: over the gradient-active adapter elements AdamW's first step
            is sign-normalised (``|dP| ~ lr0`` regardless of ``|g|``) so ``corr(|dP|,|g|) ~ 0``
            and ``median|dP| ~ lr0``; SGD's ``|dP| = c*lr0*|g|`` gives ``corr ~ 1`` and a
            median far below ``lr0``. Scale/clip-invariant: the reference chains
            ``clip_by_global_norm`` so the raw gradient is pre-scaled by
            ``c = min(1, max_grad_norm/||g||)``; the correlation/median forms survive that
            where a naive ``dP == -lr0 * g/(|g|+eps)`` identity would false-fail.

        LoRA-only (``_trainable_keys`` returns the adapter partition); returns
        ``supported: False`` otherwise so the gated tests fail cleanly on a mis-declared
        instance. Runs in the timed train worker alongside ``train_trajectory`` so one
        wall-clock kill covers a hanging reconstruction for both."""
        keep = self._trainable_keys(self.params)
        if keep is None:
            return {"supported": False, "reason": "no LoRA adapter partition (full finetuning?)"}
        adapter_names = list(keep)

        import jax
        import jax.numpy as jnp

        lr0 = self._probe_lr0()
        max_grad_norm = float(getattr(self.training_args, "max_grad_norm", 1.0) or 0.0)
        extra: dict[str, Any] = self._dpo_run_extra() if self.training_method == "dpo" else {}
        # Hand run_training the fp32 upcast and drop the native copy (as train_trajectory
        # does): a second fp32 grad on top of a resident native tree would double the
        # footprint on the 7B instances. The grad below reuses this same fp32 tree.
        run_params, native_dtypes = self._fp32_params_freeing_native()

        def _compute() -> dict[str, Any]:
            # (1) one real optimizer step through the candidate's reconstructed loop.
            result = self._recon_run_training(self.model, run_params, self.batch, self.training_args, 1, self.ignore_index, **extra)
            final = (result or {}).get("final_params")
            if final is None:
                return {"supported": False, "reason": "run_training returned no final_params"}
            in_flat = F.flatten_tree_raw(run_params)
            out_flat = F.flatten_tree_raw(final)
            if set(in_flat) != set(out_flat):
                return {"supported": False, "reason": "final_params tree structure changed"}
            adapter_set = set(adapter_names)
            # (2) base-freeze: max|delta| per leaf, one leaf at a time (no delta tree kept).
            base_max_delta = 0.0
            base_leaf_count = 0
            base_moved_leaves = 0
            adapter_max_delta = 0.0
            for name, before in in_flat.items():
                after = out_flat[name]
                d = float(jnp.max(jnp.abs(after - before))) if getattr(before, "size", 0) else 0.0
                if name in adapter_set:
                    adapter_max_delta = max(adapter_max_delta, d)
                    continue
                base_leaf_count += 1
                if d > 1e-12:
                    base_moved_leaves += 1
                base_max_delta = max(base_max_delta, d)
            # (3) step-0 gradient over the SAME adapter partition the optimizer trained.
            loss_fn = self._scalar_loss_fn()
            _lv, _tree, raw = self._value_and_grad_flat(loss_fn, run_params, keep)
            dp_parts, g_parts = [], []
            for name in adapter_names:
                dp_parts.append(np.abs(_to_np(out_flat[name] - in_flat[name]).astype(np.float64)).reshape(-1))
                g_parts.append(np.abs(_to_np(raw[name]).astype(np.float64)).reshape(-1))
            dp_all = np.concatenate(dp_parts) if dp_parts else np.zeros(0)
            g_all = np.concatenate(g_parts) if g_parts else np.zeros(0)
            gnorm = float(np.sqrt(np.sum(g_all ** 2)))
            clip = min(1.0, max_grad_norm / gnorm) if (max_grad_norm > 0 and gnorm > 0) else 1.0
            # AdamW's |dP| plateaus at lr0 where the (clipped) grad dominates eps=1e-8; pick
            # elements well above that floor so |dP|~lr0 for AdamW and |dP|=c*lr0*|g| for SGD.
            active = (clip * g_all) > 1e-6
            n_active = int(active.sum())
            if n_active >= 2:
                dp_a, g_a = dp_all[active], g_all[active]
                median_abs_dp = float(np.median(dp_a))
                if np.std(dp_a) > 0 and np.std(g_a) > 0:
                    corr = float(np.corrcoef(dp_a, g_a)[0, 1])
                else:
                    corr = 0.0  # constant |dP| (the AdamW plateau): no linear relation to |g|
            else:
                median_abs_dp = float(np.median(dp_all[dp_all > 0])) if np.any(dp_all > 0) else 0.0
                corr = 0.0
            return {
                "supported": True,
                "is_lora": True,
                "base_max_delta": base_max_delta,
                "base_leaf_count": base_leaf_count,
                "base_moved_leaves": base_moved_leaves,
                "adapter_leaf_count": len(adapter_names),
                "adapter_max_delta": adapter_max_delta,
                "adapter_moved": bool(adapter_max_delta > 0.0),
                "lr0": float(lr0),
                "grad_global_norm": gnorm,
                "clip_factor": float(clip),
                "num_active": n_active,
                "num_adapter_elems": int(dp_all.size),
                "corr_abs_dp_abs_g": corr,
                "median_abs_dp": median_abs_dp,
                "median_over_lr0": (median_abs_dp / float(lr0)) if lr0 > 0 else None,
            }

        try:
            if self.jax_device is None:
                out = _compute()
            else:
                with jax.default_device(self.jax_device):
                    out = _compute()
        except Exception as exc:  # noqa: BLE001
            return {"supported": False, "reason": f"optimizer probe failed: {type(exc).__name__}: {exc}"}
        finally:
            self._restore_native_params(run_params, native_dtypes)
        return out

    def dpo_objective_probe(self) -> dict[str, Any]:
        """Preference-sensitivity of the candidate's reconstructed DPO objective
        (``_recon_run_training``; golden-free). Swapping the chosen
        (first half) and rejected (second half) rows of the batch MUST change the step-0 loss:
        the DPO/ORPO/SimPO objectives are asymmetric in (chosen, rejected), whereas a plain
        causal-LM loss (summing all tokens) is swap-invariant. This catches a DPO repo
        that mistakenly trains a causal objective — which the train_trajectory curve can
        miss when ORPO with a small beta is numerically dominated by its SFT term.
        DPO-only; skips cleanly otherwise."""
        if self.training_method != "dpo":
            return {"supported": False, "reason": "not a DPO instance"}

        import jax.numpy as jnp

        def _swap_halves(value: Any) -> Any:
            arr = jnp.asarray(value)
            if arr.ndim >= 1 and int(arr.shape[0]) % 2 == 0 and int(arr.shape[0]) >= 2:
                half = int(arr.shape[0]) // 2
                return jnp.concatenate([arr[half:], arr[:half]], axis=0)
            return arr

        swapped = {k: _swap_halves(v) for k, v in self.batch.items()}
        # Same ref_params gate as train_trajectory: reference-free ORPO/SimPO get dpo_args
        # only (the swap-asymmetry they probe lives entirely in the policy log-ratios).
        # Built before freeing the native params (a reference-based loss would fold them in).
        kwargs: dict[str, Any] = self._dpo_run_extra()
        # Two run_training calls (base + swapped) share one fp32 upcast; freeing the native
        # copy keeps this DPO probe within the same memory budget as train_trajectory.
        run_params, native_dtypes = self._fp32_params_freeing_native()

        def _step0_loss(batch: dict[str, Any]) -> float:
            out = self._recon_run_training(self.model, run_params, batch, self.training_args, 1, self.ignore_index, **kwargs)
            losses = (out or {}).get("losses", [])
            if not losses:
                raise ValueError("run_training returned no losses")
            return float(losses[0])

        try:
            if self.jax_device is None:
                base = _step0_loss(self.batch)
                swap = _step0_loss(swapped)
            else:
                import jax

                with jax.default_device(self.jax_device):
                    base = _step0_loss(self.batch)
                    swap = _step0_loss(swapped)
        except Exception as exc:  # noqa: BLE001
            return {"supported": False, "reason": f"run_training failed: {exc}"}
        finally:
            self._restore_native_params(run_params, native_dtypes)
        return {"supported": True, "loss": base, "loss_swapped": swap, "delta": abs(base - swap)}


def _spread_indices(count: int) -> list[int]:
    """``0..count-1`` reordered so that every prefix is spread over the whole range.

    Bit-reversal permutation — deterministic, and the property that matters here is
    that truncating it (which is what a memory-capped gradient-authenticity support
    does) still samples the full depth of the network rather than whichever leaves
    happen to sort first.
    """
    if count <= 0:
        return []
    bits = max(1, (count - 1).bit_length())
    order = []
    for value in range(1 << bits):
        reversed_value = int(f"{value:0{bits}b}"[::-1], 2)
        if reversed_value < count:
            order.append(reversed_value)
    return order


def _unflatten_raw(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of ``_formulas.flatten_tree_raw`` -- rebuild the nested dict from its
    ``a/b/c`` flat names. Exact round-trip for the trees that function accepts: it
    descends dicts and stops at anything else, so anything else is a leaf here too.
    """
    tree: dict[str, Any] = {}
    for name, value in flat.items():
        *parents, last = name.split("/")
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return tree


def _to_np(value: Any) -> np.ndarray:
    """jax/np -> np.ndarray (bf16 -> fp32), torch-free."""
    if value is None:
        raise ValueError("cannot convert None to numpy")
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    return arr


# =========================================================================== #
# Subprocess plumbing.
# =========================================================================== #
def _worker_entry(conn, payload: dict[str, Any], banned: list[str]) -> None:
    import os

    try:
        os.environ[SANDBOX_ENV_FLAG] = "1"
        # Disable XLA CUDA command buffers (CUDA graphs) in this GPU worker. At
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 the XLA pool leaves almost no driver-side
        # headroom to instantiate the command buffer for this task's heavy 3B + audio-tower
        # forward, so the Omni PPO run OOMs at graph capture (RESOURCE_EXHAUSTED, "Failed to
        # instantiate CUDA graph"). Command buffers are a pure kernel-dispatch optimization,
        # so disabling them is numerically inert -- it only trades a little launch overhead
        # for the headroom the capture needs. Set before ``import jax`` so XLA reads it.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip()
        sandbox.install_torch_blocker(banned)
        # Enable float64 so the gradient-authenticity finite-difference probe is
        # numerically meaningful (a 5e8-dim unit step is below the f32 rounding
        # floor). Parity paths stay bf16 — enabling x64 only permits f64 where it
        # is explicitly requested, it does not change bf16/f32 computations.
        import jax

        jax.config.update("jax_enable_x64", True)
        runtime = _CandidateRuntime(payload)
        results = runtime.run_all()
        leaked = sandbox.banned_modules_in_sys(banned)
        conn.send({"ok": True, "results": results, "banned_modules_in_sys": leaked})
    except BaseException as exc:  # report everything to the parent
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


def _train_worker_entry(conn, payload: dict[str, Any], banned: list[str]) -> None:
    """Like ``_worker_entry`` but runs the two artifacts that drive the repo's own
    training loop -- ``train_trajectory_artifact`` (K steps) and
    ``optimizer_probe_artifact`` (1 step; base-freeze + AdamW-vs-SGD identity).

    Both live in this one spawned process so the parent can put a hard wall-clock timeout
    on ``run_training`` and kill it outright: a port that never finishes
    -- no jit, a per-step recompile, a runaway loop -- may be blocked in native XLA code
    that a Python signal cannot interrupt, so terminating the whole OS process is the only
    reliable stop, and one kill then covers both artifacts. Isolating them here means such
    a kill fails only the train tests rather than taking down every artifact in ``run_all``.
    The JAX config mirrors ``_worker_entry`` exactly (torch blocker, x64) so the trajectory
    is numerically identical to when it ran inside ``run_all`` -- the SFT/DPO golden match
    in ``test_train_trajectory`` depends on that. The result is nested
    ``{"train_trajectory": ..., "optimizer_probe": ...}``.
    """
    import os

    try:
        os.environ[SANDBOX_ENV_FLAG] = "1"
        # Disable XLA CUDA command buffers (CUDA graphs) in this GPU worker. At
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 the XLA pool leaves almost no driver-side
        # headroom to instantiate the command buffer for this task's heavy 3B + audio-tower
        # forward, so the Omni PPO run OOMs at graph capture (RESOURCE_EXHAUSTED, "Failed to
        # instantiate CUDA graph"). Command buffers are a pure kernel-dispatch optimization,
        # so disabling them is numerically inert -- it only trades a little launch overhead
        # for the headroom the capture needs. Set before ``import jax`` so XLA reads it.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip()
        sandbox.install_torch_blocker(banned)
        import jax

        jax.config.update("jax_enable_x64", True)
        runtime = _CandidateRuntime(payload)
        traj = runtime.train_trajectory_artifact()
        gc.collect()  # reclaim the K-step replay before the 1-step probe + its fp32 grad
        probe = runtime.optimizer_probe_artifact()
        result = {"train_trajectory": traj, "optimizer_probe": probe}
        leaked = sandbox.banned_modules_in_sys(banned)
        conn.send({"ok": True, "result": result, "banned_modules_in_sys": leaked})
    except BaseException as exc:  # report everything to the parent
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


def _read_real_train_metrics(output_dir: str, method: str) -> dict[str, Any]:
    """Read the metric JSONs that ``train.run_{sft,dpo,ppo}`` writes to ``output_dir`` and
    fold them into one summary consumed by the L6/T3 gates.

    The entrypoints write (via ``Trainer.save_metrics``): ``eval_pre_results.json`` (held-out
    validation loss BEFORE training), ``train_results.json`` (final train loss + step count),
    ``eval_results.json`` (held-out validation loss AFTER training). Metric key names are the
    HF convention (``{prefix}_loss``); we surface the three canonical scalars the tests read
    plus the full dicts so the T2 trend tier can reach any DPO/PPO diagnostic
    (reward margins, logps, eval_kl) without this reader having to know them all.
    """
    import json

    def _load(name: str) -> dict[str, Any]:
        p = Path(output_dir) / name
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    train_m = _load("train_results.json")
    eval_m = _load("eval_results.json")
    pre_m = _load("eval_pre_results.json")

    def _first(d: dict[str, Any], *keys: str) -> float | None:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    return None
        return None

    train_loss = _first(train_m, "train_loss", "loss")
    eval_loss = _first(eval_m, "eval_loss", "eval_token_ce", "loss")
    pre_eval_loss = _first(pre_m, "eval_pre_loss", "eval_pre_token_ce", "loss")

    return {
        # supported iff the run produced the loss the gate needs (train for T3, eval for L6);
        # a run that wrote no metrics (crashed mid-way, wrong output_dir) is a hard failure.
        "supported": train_loss is not None or eval_loss is not None,
        "method": method,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "pre_eval_loss": pre_eval_loss,
        # L6 held-out validation loss (token_ce) == the post-training eval loss.
        "V_jax": eval_loss,
        "train_metrics": train_m,
        "eval_metrics": eval_m,
        "pre_eval_metrics": pre_m,
    }


def _real_train_worker_entry(conn, payload: dict[str, Any], banned: list[str]) -> None:
    """Run the repo's REAL training entrypoint (``train.run_{sft,dpo,ppo}``) end to end,
    driven purely from the candidate's own ``config.py`` defaults, and report the metrics it
    writes. This is the in-band L6 (held-out validation loss) and T3 (end-to-end train
    run) data source: ONE real run feeds both gates.

    Unlike ``_worker_entry`` / ``_train_worker_entry`` it does NOT enable ``jax_enable_x64``:
    the real run must reproduce the standalone reference run (bf16 params, x32, highest
    matmul) against which the frozen torch/JAX bands were measured. Only ``output_dir`` is
    overridden -- to a throwaway tempdir -- so every training hyperparameter comes from
    ``config.py``, which mirrors ``torch_orig/config.py`` value-for-value (the user's
    "run real training straight from config.py" requirement). The dataset the loader reads
    (``/data``) is baked into the verifier image.
    """
    import os

    try:
        os.environ[SANDBOX_ENV_FLAG] = "1"
        # Disable XLA CUDA command buffers (CUDA graphs) in this GPU worker. At
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 the XLA pool leaves almost no driver-side
        # headroom to instantiate the command buffer for this task's heavy 3B + audio-tower
        # forward, so the Omni PPO run OOMs at graph capture (RESOURCE_EXHAUSTED, "Failed to
        # instantiate CUDA graph"). Command buffers are a pure kernel-dispatch optimization,
        # so disabling them is numerically inert -- it only trades a little launch overhead
        # for the headroom the capture needs. Set before ``import jax`` so XLA reads it.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip()
        sandbox.install_torch_blocker(banned)
        import importlib
        import sys
        import tempfile

        repo_dir = str(Path(payload["repo_dir"]).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        method = str(payload["training_method"])
        train_mod = importlib.import_module("train")
        entry = getattr(train_mod, f"run_{method}", None)
        if not callable(entry):
            conn.send({"ok": True, "result": {
                "supported": False,
                "reason": f"train.run_{method} not found in candidate repo",
            }})
            return

        with tempfile.TemporaryDirectory(prefix="t2j_realtrain_") as tmpdir:
            # Override ONLY output_dir; all hyperparameters come from config.py defaults.
            entry({"output_dir": tmpdir})
            summary = _read_real_train_metrics(tmpdir, method)

        leaked = sandbox.banned_modules_in_sys(banned)
        conn.send({"ok": True, "result": summary, "banned_modules_in_sys": leaked})
    except BaseException as exc:  # report everything to the parent
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


def _runtime_callable_param_info(fn: Any) -> dict | None:
    """Runtime analog of ``compare._callable_param_info``: the parameter shape of a live
    callable via ``inspect.signature`` -- positional slots, keyword-acceptable names, and
    whether it takes ``*args`` / ``**kwargs``. Returns the SAME dict shape the AST helper
    produces, so the parent reuses ``compare.check_callable_signature`` unchanged.

    Sees what the harness would actually hit at call time: a factory-bound closure
    (``get_train_args = build_get_train_args(...)``), a ``functools.partial``, a class
    constructor (``__init__`` minus ``self``), or any ``__call__``-able object.  None when
    the object is not callable or has no introspectable signature (a builtin / C function),
    so the caller falls back to the AST view.
    """
    import inspect

    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    positional = 0
    by_keyword: set[str] = set()
    has_var_positional = has_var_keyword = False
    for p in sig.parameters.values():
        if p.kind == p.POSITIONAL_ONLY:
            positional += 1
        elif p.kind == p.POSITIONAL_OR_KEYWORD:
            positional += 1
            by_keyword.add(p.name)
        elif p.kind == p.KEYWORD_ONLY:
            by_keyword.add(p.name)
        elif p.kind == p.VAR_POSITIONAL:
            has_var_positional = True
        elif p.kind == p.VAR_KEYWORD:
            has_var_keyword = True
    return {
        "positional_slots": positional,
        "accepts_by_keyword": sorted(by_keyword),
        "has_var_positional": has_var_positional,
        "has_var_keyword": has_var_keyword,
    }


def _runtime_constant_repr(value: Any) -> dict | None:
    """Runtime analog of ``compare._constant_repr``: the freeze-able view of a constant's
    live VALUE. A dict is reduced to its sorted string keys (``SCHEDULER_MAP`` maps names to
    function objects, ``DATA_CONFIG`` to nested structure -- only keys are comparable); a
    scalar (str / int / float / bool / None) is compared by value; anything else (an object,
    a callable) yields None so the caller falls back to the AST literal view.
    """
    if isinstance(value, dict):
        try:
            return {"kind": "dict", "keys": sorted(str(k) for k in value.keys())}
        except Exception:
            return {"kind": "dict", "keys": []}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "value", "value": value}
    return None


def _interface_worker_entry(conn, payload: dict[str, Any], banned: list[str]) -> None:
    """Import the candidate modules named in ``golden/interface.json`` and introspect their
    entry-point signatures, class method sets, and named constants at RUNTIME. Builds
    NO model and runs NO forward, so it is hardware-invariant and reports the same
    verdict with or without a GPU. Isolated in its own worker so the interface verdict never
    couples to a ``run_all`` numeric-forward crash.

    Returns plain data the parent unions with its own AST view (compare.py): for each frozen
    signature the ``inspect.signature`` param shape (or None if absent/uninspectable), for
    each frozen class its ``dir()`` method names over the MRO, for each frozen constant its
    live value/key repr, plus per-module import errors (a module that would not import is
    reported, and the parent then falls back to AST for it).
    """
    import os

    try:
        os.environ[SANDBOX_ENV_FLAG] = "1"
        sandbox.install_torch_blocker(banned)
        import importlib
        import sys as _sys

        repo_dir = str(Path(payload["repo_dir"]).resolve())
        if repo_dir not in _sys.path:
            _sys.path.insert(0, repo_dir)
        spec = payload.get("interface_spec") or {}

        def _module_name(rel: str) -> str:
            rel = rel[:-3] if rel.endswith(".py") else rel
            return rel.replace("/", ".").replace("\\", ".")

        import_errors: dict[str, str] = {}
        module_cache: dict[str, Any] = {}

        def _get_module(rel: str):
            name = _module_name(rel)
            if name in module_cache:
                return module_cache[name]
            try:
                mod = importlib.import_module(name)
            except BaseException as exc:  # a contract module that will not import in isolation
                import_errors[rel] = f"{type(exc).__name__}: {exc}"
                mod = None
            module_cache[name] = mod
            return mod

        # --- signatures: does the live callable accept the harness call shape? ---------- #
        signatures: dict[str, Any] = {}
        for ref in (spec.get("signatures") or {}):
            rel, _, name = ref.partition("::")
            mod = _get_module(rel)
            info = None
            if mod is not None:
                fn = getattr(mod, name, None)
                if fn is not None:
                    info = _runtime_callable_param_info(fn)
            signatures[ref] = info

        # --- methods: the class's non-underscore method set over the MRO (dir()) -------- #
        methods: dict[str, dict[str, list[str]]] = {}
        for rel, classes in (spec.get("methods") or {}).items():
            mod = _get_module(rel)
            rel_out: dict[str, list[str]] = {}
            if mod is not None:
                for cls_name in classes:
                    cls = getattr(mod, cls_name, None)
                    if isinstance(cls, type):
                        rel_out[cls_name] = sorted(
                            n for n in dir(cls) if not n.startswith("_")
                        )
            methods[rel] = rel_out

        # --- constants: the live value / key set on the SAME module --------------------- #
        constants: dict[str, dict[str, Any]] = {}
        for rel, names in (spec.get("constants") or {}).items():
            mod = _get_module(rel)
            rel_out: dict[str, Any] = {}
            if mod is not None:
                for const_name in names:
                    if hasattr(mod, const_name):
                        rep = _runtime_constant_repr(getattr(mod, const_name))
                        if rep is not None:
                            rel_out[const_name] = rep
            constants[rel] = rel_out

        result = {
            "supported": True,
            "signatures": signatures,
            "methods": methods,
            "constants": constants,
            "import_errors": import_errors,
        }
        leaked = sandbox.banned_modules_in_sys(banned)
        conn.send({"ok": True, "result": result, "banned_modules_in_sys": leaked})
    except BaseException as exc:  # report everything to the parent
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


# =========================================================================== #
# Parent-side handle.
# =========================================================================== #
class Candidate:
    """Parent-side handle: spawns the torch-blocked worker once and caches results."""

    def __init__(
        self,
        repo_dir: str,
        *,
        inputs: dict[str, np.ndarray],
        repo_config: dict[str, Any],
        training_method: str,
        seed: int = 42,
        loss_trajectory_steps: int = 3,
        generation_max_new_tokens: int = 16,
        generation_sequences: np.ndarray | None = None,
        battery_inputs: dict[str, dict[str, np.ndarray]] | None = None,
        compare_hidden_states: bool = False,
        device: str | None = None,
        banned: list[str] | None = None,
        timeout: float = 1800.0,
        train_timeout_s: float = 1800.0,
    ) -> None:
        self.repo_dir = str(Path(repo_dir).resolve())
        self._banned = list(banned) if banned is not None else list(sandbox.DEFAULT_BANNED_MODULES)
        # No weights are passed in. repo_config["model_name_or_path"] is the HF repo id;
        # the candidate's loader resolves it to /data/{repo_id} and converts the
        # checkpoint itself.
        self._payload = {
            "repo_dir": self.repo_dir,
            "repo_config": dict(repo_config),
            "training_method": training_method,
            "inputs": {k: np.asarray(v) for k, v in inputs.items()},
            "seed": seed,
            "loss_trajectory_steps": loss_trajectory_steps,
            "generation_max_new_tokens": generation_max_new_tokens,
            # Reference's frozen generated sequence for the teacher-forced probe
            # (None for DPO/PPO, which freeze no generation).
            "generation_sequences": None if generation_sequences is None else np.asarray(generation_sequences),
            # Per-variant frozen inputs for the input-space battery (None for
            # multimodal / older goldens that froze no battery).
            "battery_inputs": None if battery_inputs is None else {
                name: {k: np.asarray(v) for k, v in inp.items()} for name, inp in battery_inputs.items()
            },
            "compare_hidden_states": compare_hidden_states,
            "device": device,
        }
        self._timeout = timeout
        # A separate, generous wall-clock budget for the repo's OWN training loop,
        # which runs in its own worker (see _ensure_train_run). Per instance
        # in golden/meta.json ("train_timeout_s").
        self._train_timeout_s = float(train_timeout_s)
        self._results: dict[str, Any] | None = None
        self._error: dict[str, Any] | None = None
        self._train_result: dict[str, Any] | None = None
        # The single REAL end-to-end training run (train.run_{sft,dpo,ppo} from config.py
        # defaults) that feeds BOTH the in-band L6 (validation_loss) and T3
        # (train_run) behavioral gates -- run once, cached, read twice.
        self._real_train_result: dict[str, Any] | None = None
        # Runtime interface introspection (signatures/methods/constants),
        # imported+introspected in its own torch-blocked worker independently of any
        # GPU forward so the interface verdict stays hardware-invariant. Cached;
        # read by test_interface_signature / _methods / test_constant_values.
        self._interface_result: dict[str, Any] | None = None

    # --- execution -------------------------------------------------------- #
    def _ensure_run(self) -> dict[str, Any]:
        if self._results is not None:
            return self._results
        if self._error is not None:
            self._raise_error()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker_entry, args=(child_conn, self._payload, self._banned))
        process.start()
        child_conn.close()
        if parent_conn.poll(self._timeout):
            try:
                payload = parent_conn.recv()
            except EOFError:
                payload = {"ok": False, "error": "candidate sandbox closed without sending a result"}
        else:
            process.terminate()
            payload = {"ok": False, "error": f"candidate sandbox timed out after {self._timeout}s"}
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
        if payload.get("ok") and payload.get("banned_modules_in_sys"):
            payload = {
                "ok": False,
                "error": f"candidate imported banned modules inside the sandbox: {payload['banned_modules_in_sys']}",
            }
        if not payload.get("ok"):
            self._error = payload
            self._raise_error()
        self._results = payload["results"]
        return self._results

    def _raise_error(self) -> None:
        assert self._error is not None
        detail = self._error.get("traceback") or self._error.get("error") or "unknown candidate failure"
        raise CandidateError(f"candidate sandbox failed:\n{detail}")

    @staticmethod
    def _hard_kill(process) -> None:
        """SIGTERM, then SIGKILL if it will not die within the grace period. A runaway XLA
        compile can sit in native code that ignores SIGTERM; escalating to SIGKILL keeps a
        timed-out (or already-finished) train worker -- and the GPU memory it holds -- from
        being orphaned."""
        if process.is_alive():
            process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

    def _ensure_train_run(self) -> dict[str, Any]:
        """Run the repo's own training loop (reconstructed) in a dedicated spawned worker under
        a hard wall-clock timeout (``train_timeout_s``), and cache the result.

        Returns ``{"train_trajectory": {...}, "optimizer_probe": {...}}`` -- the K-step
        trajectory artifact (``{"supported": bool, "losses": [...]}``) and the 1-step
        base-freeze/optimizer-identity probe. On timeout BOTH sub-results are
        ``{"supported": False, "timed_out": True, ...}`` so their tests fail while every
        other test -- served from the separate ``run_all`` worker -- is
        unaffected. A failure to even build the runtime (e.g. the model will not load) is a
        hard error, exactly as in ``_ensure_run``; a ``run_training`` error is caught inside
        each artifact and surfaces as ``supported: False``.
        """
        if self._train_result is not None:
            return self._train_result
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_train_worker_entry, args=(child_conn, self._payload, self._banned))
        process.start()
        child_conn.close()
        payload: dict[str, Any] | None = None
        if parent_conn.poll(self._train_timeout_s):
            try:
                payload = parent_conn.recv()
            except EOFError:
                payload = {"ok": False, "error": "candidate train sandbox closed without sending a result"}
        # Always reap the worker: on timeout to kill the runaway loop, on success to reclaim
        # its GPU memory before the next test.
        self._hard_kill(process)
        if payload is None:  # nothing arrived within the budget -> timed out
            timed_out = {
                "supported": False,
                "timed_out": True,
                "reason": f"training loop exceeded train_timeout_s={self._train_timeout_s:.0f}s",
            }
            self._train_result = {"train_trajectory": timed_out, "optimizer_probe": dict(timed_out)}
            return self._train_result
        if payload.get("ok") and payload.get("banned_modules_in_sys"):
            payload = {
                "ok": False,
                "error": f"candidate imported banned modules inside the sandbox: {payload['banned_modules_in_sys']}",
            }
        if not payload.get("ok"):
            self._error = payload
            self._raise_error()
        self._train_result = payload["result"]
        return self._train_result

    def _ensure_real_train_run(self) -> dict[str, Any]:
        """Run the repo's REAL training entrypoint once, under the hard ``train_timeout_s``
        wall-clock budget, and cache it. Feeds BOTH the in-band L6 (``validation_loss``)
        and T3 (``train_run``) gates -- one run, two readers.

        On timeout the whole worker is killed and ``{"supported": False, "timed_out": True}``
        is returned so BOTH gates fail: a JAX port that cannot reproduce the reference within
        the calibrated runtime budget fails the runtime dimension by design (the timeout is
        pinned to the PyTorch reference runtime). A failure to import/launch ``train`` (or a
        banned-module leak) is a hard error, exactly as in ``_ensure_run``; an in-run
        exception surfaces as the worker's ``ok: False`` and is also raised.
        """
        if self._real_train_result is not None:
            return self._real_train_result
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_real_train_worker_entry, args=(child_conn, self._payload, self._banned))
        process.start()
        child_conn.close()
        payload: dict[str, Any] | None = None
        if parent_conn.poll(self._train_timeout_s):
            try:
                payload = parent_conn.recv()
            except EOFError:
                payload = {"ok": False, "error": "candidate real-train sandbox closed without sending a result"}
        # Always reap the worker: on timeout to kill the runaway run, on success to reclaim
        # its GPU memory before the next test.
        self._hard_kill(process)
        if payload is None:  # nothing arrived within the budget -> timed out
            self._real_train_result = {
                "supported": False,
                "timed_out": True,
                "reason": f"real training run exceeded train_timeout_s={self._train_timeout_s:.0f}s",
            }
            return self._real_train_result
        if payload.get("ok") and payload.get("banned_modules_in_sys"):
            payload = {
                "ok": False,
                "error": f"candidate imported banned modules inside the sandbox: {payload['banned_modules_in_sys']}",
            }
        if not payload.get("ok"):
            self._error = payload
            self._raise_error()
        self._real_train_result = payload["result"]
        return self._real_train_result

    def _ensure_interface(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Spawn the torch-blocked interface worker once (importing + introspecting the
        contract modules named in ``spec`` == golden/interface.json) and cache its data.

        The verdict is decoupled from any forward run: on a worker crash / timeout / banned
        leak (``test_no_torch`` reports leaks separately) this returns ``supported=False``,
        and the interface tests fall back to their AST view -- so the interface stage stays
        hardware-invariant and no worse than the pure-static gate it augments. ``spec``
        is constant for the instance, so the first probe is reused by every interface test.
        """
        if self._interface_result is not None:
            return self._interface_result
        payload = {
            "repo_dir": self.repo_dir,
            "interface_spec": {
                "signatures": spec.get("signatures") or {},
                "methods": spec.get("methods") or {},
                "constants": spec.get("constants") or {},
            },
        }
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_interface_worker_entry, args=(child_conn, payload, self._banned))
        process.start()
        child_conn.close()
        result_payload: dict[str, Any] | None = None
        if parent_conn.poll(self._timeout):
            try:
                result_payload = parent_conn.recv()
            except EOFError:
                result_payload = {"ok": False, "error": "interface sandbox closed without a result"}
        self._hard_kill(process)
        if result_payload is None:  # nothing arrived within the budget -> AST fallback
            self._interface_result = {
                "supported": False,
                "reason": f"interface probe exceeded {self._timeout:.0f}s",
            }
            return self._interface_result
        if result_payload.get("ok") and result_payload.get("banned_modules_in_sys"):
            # A banned import is a real violation, reported by test_no_torch; here it just
            # disables the runtime view so the interface tests use their AST fallback.
            self._interface_result = {
                "supported": False,
                "reason": f"candidate imported banned modules inside the sandbox: {result_payload['banned_modules_in_sys']}",
            }
            return self._interface_result
        if not result_payload.get("ok"):
            # Do NOT couple the interface verdict to a probe crash: fall back to AST.
            self._interface_result = {
                "supported": False,
                "reason": result_payload.get("error") or "interface probe failed",
                "traceback": result_payload.get("traceback"),
            }
            return self._interface_result
        self._interface_result = result_payload["result"]
        return self._interface_result

    def interface(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Runtime interface introspection against ``spec`` (golden/interface.json): the live
        signature / method / constant view the parent unions with its AST view.
        ``supported=False`` means the probe could not run and the caller should use AST only.
        """
        return self._ensure_interface(spec)

    # --- parent-only static scan ----------------------------------------- #
    def purity_scan(self) -> purity.PurityReport:
        """Static AST scan of the whole candidate repo (no convert/ exception)."""
        return purity.scan_repo_for_torch(self.repo_dir, allow_dirs=())

    # --- cached accessors ------------------------------------------------- #
    def param_spec(self) -> dict[str, Any]:
        return self._ensure_run()["param_spec"]

    def param_norms(self) -> dict[str, Any]:
        return self._ensure_run()["param_norms"]

    def structural(self) -> dict[str, Any]:
        return self._ensure_run()["structural"]

    def forward_purity(self) -> dict[str, Any]:
        return self._ensure_run()["forward_purity"]

    def forward(self) -> dict[str, Any]:
        return self._ensure_run()["forward"]

    def loss(self) -> dict[str, Any]:
        return self._ensure_run()["loss_artifact"]

    def batch_schema(self) -> dict[str, Any]:
        return self._ensure_run()["batch_schema"]

    def batch_arrays(self) -> dict[str, Any]:
        return self._ensure_run().get("batch_arrays", {"supported": False, "reason": "not run"})

    def gradients(self) -> dict[str, Any]:
        return self._ensure_run()["gradient"]

    def gradient_authenticity(self) -> dict[str, Any]:
        return self._ensure_run()["gradient_authenticity"]

    def lr_schedule(self) -> dict[str, Any]:
        return self._ensure_run()["lr_schedule"]

    def loss_trajectory(self) -> dict[str, Any]:
        return self._ensure_run()["loss_trajectory"]

    def train_trajectory(self) -> dict[str, Any]:
        # Its own timed worker, NOT the shared run_all results.
        return self._ensure_train_run().get(
            "train_trajectory", {"supported": False, "reason": "not run"}
        )

    def optimizer_probe(self) -> dict[str, Any]:
        # LoRA base-freeze + AdamW-vs-SGD identity; same timed worker as
        # train_trajectory, so a hung run_training kills both at once.
        return self._ensure_train_run().get(
            "optimizer_probe", {"supported": False, "reason": "not run"}
        )

    def dpo_objective(self) -> dict[str, Any]:
        return self._ensure_run().get("dpo_objective", {"supported": False, "reason": "not run"})

    def determinism(self) -> dict[str, Any]:
        return self._ensure_run()["determinism"]

    def generation(self) -> dict[str, Any]:
        return self._ensure_run().get("generation", {"supported": False, "reason": "not an SFT instance"})

    def generation_logits(self) -> dict[str, Any]:
        # Teacher-forced continuation-logit parity; SFT-only, text-only.
        return self._ensure_run().get(
            "generation_logits", {"supported": False, "reason": "not an SFT instance"}
        )

    def self_consistency(self) -> dict[str, Any]:
        return self._ensure_run().get("self_consistency", {"supported": False, "reason": "not run"})

    def mm_tower_sensitivity(self) -> dict[str, Any]:
        # Vision-tower sensitivity gate (golden-free {0,1}); multimodal only.
        return self._ensure_run().get(
            "mm_tower_sensitivity", {"supported": False, "reason": "not a multimodal instance"}
        )

    def battery(self) -> dict[str, Any]:
        # Input-space generality battery; text-only, deselected on goldens with no battery.
        return self._ensure_run().get("battery", {"supported": False, "reason": "not run"})

    def fp32_forward(self) -> dict[str, Any]:
        return self._ensure_run().get("fp32_forward", {"supported": False, "reason": "not run"})

    def config_spec(self) -> dict[str, Any]:
        # Resolved training-config parity (test_config_spec); presence-gated golden.
        return self._ensure_run().get("config_spec", {"supported": False, "reason": "not run"})

    def loader_spec(self) -> dict[str, Any]:
        # Tokenizer-fact parity (test_loader_spec); presence-gated golden.
        return self._ensure_run().get("loader_spec", {"supported": False, "reason": "not run"})

    def validation_split(self) -> dict[str, Any]:
        """L6 held-out validation split boundary (test_validation_split).

        The trainer-side determinism (seeded shuffle + fp32/deterministic eval) is done; the
        grade-side split-boundary probe is still pending. No golden freezes
        ``validation_loss.json`` yet, so the test is deselected -- this returns a pending
        marker (WITHOUT spawning the sandbox) so that if such a golden ever appears the test
        FAILS loudly rather than silently passing until the probe is implemented.
        """
        return {"supported": False, "reason": "L6 held-out validation split probe pending"}

    def validation_loss(self) -> dict[str, Any]:
        """L6 held-out validation loss (test_validation_loss). The post-training
        ``eval_loss`` (``V_jax``) of the real end-to-end training run -- one shared run,
        also read by ``train_run`` for T3."""
        return self._ensure_real_train_run()

    def train_run(self) -> dict[str, Any]:
        """T3 end-to-end train run (test_train_run_end_to_end). The same cached real run
        as ``validation_loss``: train/eval/pre-eval losses (+ any DPO/PPO trend diagnostics
        in ``eval_metrics``) from ``train.run_{sft,dpo,ppo}`` driven by ``config.py``."""
        return self._ensure_real_train_run()
