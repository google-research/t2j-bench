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

"""Helpers for the per-submodule parity tests (Route B), torch-free.

The ``@refunits`` tests import the candidate repo directly (``T2J_JAX_REPO`` or
``../solution/ref_jax_repo``) and compare each critical submodule/function to the torch
reference frozen in ``golden/reference_units.{npz,json}`` from the reference
implementation. Comparisons use the same ``_t2j.compare`` gates
+ ``golden/meta.json`` thresholds as the rest of the grader. Everything runs in float32 (the
JAX side upcasts params) so the checks isolate each unit's math from bf16/fp16 kernel noise.

This is the GENERIC grade side: a single fp32 forward with flax ``capture_intermediates=True``
exposes every named submodule's output BY PATH (identical to the torch attribute path the freeze
side hooked), so no per-family module reconstruction is needed. The list of frozen submodule
paths is read back from ``reference_units.json['modeling_units']`` -- this file is byte-identical
across every text family; only the frozen goldens differ.

The candidate model is built through the DOCUMENTED loader contract ALONE -- ``config.get_train_args``
-> ``model.loader.load_tokenizer`` -> ``model.loader.load_model`` -- exactly as ``_t2j/candidate.py``
builds it, never by reaching into a ref-only ``transformers_impl.modeling`` /
``transformers_impl.configuration`` / private ``_resolve_model_path``. A faithful port that
"mirrors torch_orig's structure and public API" (naming its own) exposes ``load_model`` but need
not expose those internal module names; grading its forward through the contract therefore grades
the PORT, not the reference layout. The DPO trainer units are likewise reconstructed grader-side
from the forward + vendored ``_t2j._formulas`` math (:class:`_ReconDPOTrainer`) instead of
importing ``dpo.trainer`` -- byte-faithful to the reference ``DPOTrainer`` methods, so the frozen
torch goldens still score ~1.0 for a correct port.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_TEST_DIR = Path(__file__).resolve().parent
_GOLDEN = _TEST_DIR / "golden"


def repo_dir() -> str:
    d = os.environ.get("T2J_JAX_REPO") or str((_TEST_DIR / ".." / "solution" / "ref_jax_repo").resolve())
    d = str(Path(d).resolve())
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def _cand_symbol(module_name: str, attr: str):
    """Lever-A soft-import: the candidate repo's ``module_name.attr`` (e.g. the port's own
    ``dpo.trainer.DPOTrainer``), or ``None`` if the port does not expose it / it cannot be
    imported. Never raises -- a missing symbol means 'no objective discrimination for this
    port', NOT a false failure of a faithful port that organises its objective differently."""
    try:
        repo_dir()
        import importlib

        return getattr(importlib.import_module(module_name), attr, None)
    except Exception:
        return None


def units_present() -> bool:
    return (_GOLDEN / "reference_units.npz").exists() and (_GOLDEN / "reference_units.json").exists()


@lru_cache(maxsize=1)
def arrays() -> dict[str, np.ndarray]:
    return {k: v for k, v in np.load(_GOLDEN / "reference_units.npz").items()}


@lru_cache(maxsize=1)
def scalars() -> dict[str, Any]:
    return json.loads((_GOLDEN / "reference_units.json").read_text())


def modeling_units() -> list[str]:
    """The logical submodule paths the freeze side captured (e.g. ``model/layers_0/self_attn``)."""
    if not units_present():
        return []
    return list(scalars().get("modeling_units", []))


@lru_cache(maxsize=1)
def meta() -> dict[str, Any]:
    return json.loads((_GOLDEN / "meta.json").read_text())


@lru_cache(maxsize=1)
def repo_config() -> dict[str, Any]:
    return dict(meta()["repo_config"])


@lru_cache(maxsize=1)
def train_args_tuple():
    """(model_args, data_args, training_args, finetuning_args, dpo_args) from the repo."""
    repo_dir()
    import config as C

    return C.get_train_args(repo_config())


# --------------------------------------------------------------------------- #
# Candidate model, built through the DOCUMENTED loader contract only.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _loaded():
    """``(module, params_f32)`` from the candidate's own ``model.loader.load_model`` -- the
    SAME contract path ``_t2j/candidate.py`` uses (``config.get_train_args`` -> ``load_tokenizer``
    -> ``load_model``), never a ref-only modeling/configuration import. Params are upcast to
    float32 for tight parity (as the old ``convert_checkpoint`` path did). ``is_trainable=True``
    means the LoRA adapter leaves are present but zero-initialised (B=0), a no-op on this
    step-0 forward, so both the captured intermediates and the DPO log-probs are byte-identical
    to a bare converted checkpoint -- and the capture paths are LoRA-invariant (the adapter is
    param leaves, not a submodule wrapper)."""
    repo_dir()
    import importlib
    import inspect

    import jax
    import jax.numpy as jnp

    args = train_args_tuple()
    model_args, finetuning_args = args[0], args[3]
    loader = importlib.import_module("model.loader")

    tok = loader.load_tokenizer(model_args)
    tokenizer = tok["tokenizer"] if isinstance(tok, dict) else tok

    kwargs: dict[str, Any] = {
        "tokenizer": tokenizer,
        "model_args": model_args,
        "finetuning_args": finetuning_args,
        "is_trainable": True,
    }
    if str(meta().get("training_method")) == "ppo":
        # The value head is in no checkpoint; the caller asks for it, exactly as candidate.py.
        kwargs["add_valuehead"] = True
    sig = inspect.signature(loader.load_model)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    accepted = {k: v for k, v in kwargs.items() if has_var_kw or k in sig.parameters}
    loaded = loader.load_model(**accepted)

    if isinstance(loaded, tuple):
        module = loaded[0]
        params = loaded[1] if len(loaded) > 1 else None
    else:
        module = getattr(loaded, "module", loaded)
        params = getattr(loaded, "params", None)
    if params is None:
        raise ValueError("candidate load_model did not expose a JAX params pytree")
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
    return module, params


def jax_module():
    """The candidate flax module (fp32 compute), built via the loader contract."""
    return _loaded()[0]


@lru_cache(maxsize=1)
def params_f32():
    """The candidate param pytree, in float32 (see :func:`_loaded`)."""
    return _loaded()[1]


def a(name: str) -> np.ndarray:
    """Frozen reference array by logical name. Modeling units are stored under a '.'-joined
    key (npz keys cannot carry '/'), so a slash path resolves to its sanitized key too."""
    arr = arrays()
    if name in arr:
        key = name
    elif name.replace("/", ".") in arr:
        key = name.replace("/", ".")
    else:
        raise KeyError(name)
    return np.asarray(arr[key], dtype=np.float32)


@lru_cache(maxsize=1)
def golden_inputs() -> dict[str, np.ndarray]:
    """The frozen benchmark batch (input_ids / attention_mask / labels)."""
    return {k: v for k, v in np.load(_GOLDEN / "inputs.npz").items()}


def sub(params, *path):
    """Descend into the param pytree (e.g. sub(params, 'model', 'layers_0', 'self_attn'))."""
    cur = params
    for key in path:
        cur = cur[key]
    return cur


# --------------------------------------------------------------------------- #
# Shared comparison helpers for the per-unit parity tests.
# --------------------------------------------------------------------------- #
def jnp(x):
    import jax.numpy as _jnp

    return _jnp.asarray(x)


def sim_array(name: str, got) -> float:
    """Continuous ``S_tensor`` similarity in [0,1] of ``got`` vs the frozen reference array
    ``name``. Both direction AND magnitude must agree; a shape
    mismatch / non-finite candidate / missing reference key -> 0.0 (never raises)."""
    from _t2j import similarity

    try:
        ref = a(name)
    except KeyError:
        return 0.0
    return similarity.S_tensor(ref, got)


def sim_scalar(ref, got) -> float:
    """Continuous ``S_scalar`` similarity for an isolated pure-math unit scalar."""
    from _t2j import similarity

    return similarity.S_scalar(ref, got)


# --------------------------------------------------------------------------- #
# JAX forward-with-intermediates (single fp32 capture).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def jax_forward_capture():
    """Run ONE fp32 forward of the candidate ForCausalLM with flax
    ``capture_intermediates=True``; return ``(output, intermediates_tree)``. Every named
    submodule's output is captured by path (mirroring the torch attribute path)."""
    model = jax_module()
    inp = golden_inputs()
    out, state = model.apply(
        {"params": params_f32()},
        jnp(inp["input_ids"]), jnp(inp["attention_mask"]),
        output_hidden_states=True,
        capture_intermediates=True, mutable=["intermediates"],
    )
    return out, state["intermediates"]


def intermediate(path: str):
    """Captured output for a logical slash path (e.g. ``model/layers_0/self_attn``).

    flax ``sow`` wraps each captured value in a 1-tuple; a submodule that itself returns a
    tuple (e.g. attention) is unwrapped to its first element to mirror the torch hook."""
    _out, inter = jax_forward_capture()
    node = inter
    for comp in path.split("/"):
        node = node[comp]
    val = node["__call__"]
    if isinstance(val, tuple):
        val = val[0]
    if isinstance(val, tuple):
        val = val[0]
    return val


def jax_batch():
    """The frozen benchmark batch as jax arrays (input_ids/attention_mask/labels)."""
    inp = golden_inputs()
    return {
        "input_ids": jnp(inp["input_ids"]),
        "attention_mask": jnp(inp["attention_mask"]),
        "labels": jnp(inp["labels"]),
    }


# --------------------------------------------------------------------------- #
# DPO trainer units, reconstructed grader-side (contract + vendored math only).
# --------------------------------------------------------------------------- #
def get_batch_logps(logits, labels, ignore_index: int = -100):
    """Per-example summed log-prob of the labels + valid-token count.

    Lever-A: PREFER the candidate's OWN ``dpo.trainer.get_batch_logps`` so a wrong label
    mask / log-prob reduction is scored against the port's real code; SOFT FALLBACK to the
    vendored ``_t2j._formulas.jax_batch_logps`` (identical math, same ``(logps, valid_length)``
    contract) if the port does not expose it -- a faithful port that organises this differently
    is not false-failed, it simply earns no discrimination here."""
    fn = _cand_symbol("dpo.trainer", "get_batch_logps")
    if fn is not None:
        try:
            return fn(logits, labels, ignore_index)
        except Exception:
            pass
    from _t2j import _formulas as F

    return F.jax_batch_logps(logits, labels, ignore_index)


class _ReconDPOTrainer:
    """Grader-side reconstruction of the reference ``DPOTrainer``'s SCORED units, built from the
    documented forward (``module.apply``) + vendored ``_t2j._formulas`` math ALONE -- never the
    candidate's ``dpo.trainer``. Every method is byte-faithful to the reference ``DPOTrainer``
    (``concatenated_forward`` / ``orpo_loss`` / ``simpo_loss`` / ``dpo_loss`` /
    ``get_batch_loss_metrics``), so the frozen torch goldens score identically to the old
    ``dpo.trainer``-backed path for a correct port."""

    IGNORE_INDEX = -100

    def __init__(self, module, dpo_args):
        self.module = module
        self.beta = float(getattr(dpo_args, "pref_beta", 0.1))
        self.loss_type = str(getattr(dpo_args, "pref_loss", "sigmoid"))
        self.label_smoothing = float(getattr(dpo_args, "dpo_label_smoothing", 0.0))
        self.simpo_gamma = float(getattr(dpo_args, "simpo_gamma", 0.5))
        self.ftx_gamma = float(getattr(dpo_args, "pref_ftx", 0.0))
        self.ignore_index = self.IGNORE_INDEX
        self.use_ref_model = self.loss_type not in ("orpo", "simpo")

    def concatenated_forward(self, params, batch):
        from _t2j import _formulas as F

        labels = batch["labels"]
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        all_logits = self.module.apply({"params": params}, **model_inputs).logits
        all_logps, valid_length = F.jax_batch_logps(all_logits, labels, self.ignore_index)
        if self.loss_type in ("orpo", "simpo"):
            all_logps_normalized = all_logps / valid_length
        else:
            all_logps_normalized = all_logps
        batch_size = model_inputs["input_ids"].shape[0] // 2
        chosen_logps, rejected_logps = all_logps_normalized[:batch_size], all_logps_normalized[batch_size:]
        chosen_logits, rejected_logits = all_logits[:batch_size], all_logits[batch_size:]
        chosen_length = valid_length[:batch_size]
        if self.loss_type in ("orpo", "simpo"):
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = all_logps[:batch_size] / chosen_length
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
        }

    def orpo_loss(self, chosen_logps, rejected_logps):
        import jax
        import jax.numpy as _jnp

        log_odds = (chosen_logps - rejected_logps) - (
            _jnp.log1p(-_jnp.exp(chosen_logps)) - _jnp.log1p(-_jnp.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -jax.nn.log_sigmoid(log_odds)
        return sft_loss + self.beta * odds_ratio_loss

    def simpo_loss(self, chosen_logps, rejected_logps):
        import jax

        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        return -jax.nn.log_sigmoid(self.beta * logits)

    def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
        import jax

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_rewards = self.beta * jax.lax.stop_gradient(chosen_logratios)
        rejected_rewards = self.beta * jax.lax.stop_gradient(rejected_logratios)
        logits = chosen_logratios - rejected_logratios
        if self.label_smoothing > 0:
            losses = (
                -jax.nn.log_sigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - jax.nn.log_sigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            losses = -jax.nn.log_sigmoid(self.beta * logits)
        return losses, chosen_rewards, rejected_rewards

    def _compute_reference_log_probs(self, params, batch):
        import jax

        ref_params = jax.lax.stop_gradient(params)
        ref_out = self.concatenated_forward(ref_params, batch)
        return (
            jax.lax.stop_gradient(ref_out["chosen_logps"]),
            jax.lax.stop_gradient(ref_out["rejected_logps"]),
        )

    def get_batch_loss_metrics(self, params, batch, train_eval: str = "train"):
        import jax
        import jax.numpy as _jnp

        out = self.concatenated_forward(params, batch)
        policy_chosen_logps = out["chosen_logps"]
        policy_rejected_logps = out["rejected_logps"]
        if self.use_ref_model:
            ref_chosen_logps, ref_rejected_logps = self._compute_reference_log_probs(params, batch)
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
        else:
            if self.loss_type == "orpo":
                losses = self.orpo_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
            chosen_rewards = self.beta * jax.lax.stop_gradient(policy_chosen_logps)
            rejected_rewards = self.beta * jax.lax.stop_gradient(policy_rejected_logps)
        if self.ftx_gamma > 1e-6:
            losses = losses + self.ftx_gamma * (-out["chosen_logps_avg"])
        loss = _jnp.mean(losses)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}rewards/chosen": float(_jnp.mean(chosen_rewards)),
            f"{prefix}rewards/rejected": float(_jnp.mean(rejected_rewards)),
            f"{prefix}rewards/accuracies": float(_jnp.mean((chosen_rewards > rejected_rewards).astype(_jnp.float32))),
            f"{prefix}rewards/margins": float(_jnp.mean(chosen_rewards - rejected_rewards)),
            f"{prefix}logps/chosen": float(_jnp.mean(policy_chosen_logps)),
            f"{prefix}logps/rejected": float(_jnp.mean(policy_rejected_logps)),
            f"{prefix}logits/chosen": float(_jnp.mean(out["chosen_logits"])),
            f"{prefix}logits/rejected": float(_jnp.mean(out["rejected_logits"])),
        }
        return loss, metrics


def jax_trainer():
    """(grader-side DPO trainer shim over the candidate module/params, dpo_args).

    Reconstructed from the documented forward + vendored math (:class:`_ReconDPOTrainer`),
    NOT ``dpo.trainer.DPOTrainer`` -- a faithful port need not expose that class, and the JAX
    construction convention is a ref_jax detail the grader must not depend on. The shim's
    methods are byte-faithful to the reference trainer, so the frozen torch goldens still score
    ~1.0. (The module-level ``get_batch_logps`` above IS re-pointed to the candidate's own,
    since that symbol is torch_orig-grounded and takes frozen logits -- no trainer needed.)"""
    _m, _d, _training_args, _f, dpo_args = train_args_tuple()
    return _ReconDPOTrainer(jax_module(), dpo_args), dpo_args
