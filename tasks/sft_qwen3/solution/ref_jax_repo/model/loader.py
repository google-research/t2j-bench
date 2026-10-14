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

"""Tokenizer + native-flax model loading — mirror of ``torch_orig/model/loader.py``.

Same public API: ``TokenizerModule``, ``load_tokenizer``, ``load_model``,
``load_ref_model``. JAX adjustments:
- ``load_model`` returns ``(module, params, frozen_prefixes)`` — the flax module +
  parameter pytree — instead of a stateful ``nn.Module`` (torch's return). The weights
  come from ``convert_checkpoint`` below, which reads the very same Hugging Face
  checkpoint torch reads (``/data/{repo_id}``) and remaps it onto the flax pytree.
- ``load_ref_model`` returns ``(module, params, [])`` (or ``None`` for reference-free
  ORPO/SimPO), the frozen reference policy; the trainer takes its log-probs under
  ``jax.lax.stop_gradient``.
- The model class + config are resolved from the local ``transformers_impl`` package
  (``Qwen3ForCausalLM`` / ``config_from_pretrained``), mirroring torch's
  ``_resolve_local_model_class`` / ``_load_local_config``.
"""

from __future__ import annotations

import dataclasses as _dc
import json
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp

from transformers_impl.configuration import Qwen3Config, config_from_pretrained
from transformers_impl.modeling import Qwen3ForCausalLM, lora_a_init

_LOCAL_TRANSFORMERS_PACKAGE = "transformers_impl"
_DEFAULT_CONFIG_CLASS = "Qwen3Config"
_DEFAULT_ARCHITECTURE = "Qwen3ForCausalLM"


class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


_MODEL_ROOT = Path("/data")


def _resolve_model_path(model_name_or_path: str) -> Path:
    """Resolve a model id to the directory its tokenizer/config were baked into.

    The verifier image carries the whole checkpoint at /data/{repo_id} — tokenizer,
    config, and the safetensors ``convert_checkpoint`` reads — and runs under no-network,
    so an id that is not on disk is a hard error, never a download.
    """
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return candidate.resolve()
    local = _MODEL_ROOT / model_name_or_path
    if local.is_dir():
        return local
    raise FileNotFoundError(
        f"Model {model_name_or_path!r} not found at {local}. The tokenizer/config files "
        "are baked into the image at build time; nothing is fetched at run time."
    )


# --------------------------------------------------------------------------- #
# Weight conversion (JAX adjustment of torch's _load_state_dict + load_state_dict).
# --------------------------------------------------------------------------- #
def _checkpoint_files(model_path: Path) -> list[Path]:
    """The checkpoint shard(s) to read — mirror of torch's ``_checkpoint_files``."""
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [model_path / shard for shard in sorted(set(weight_map.values()))]
    single = model_path / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"No .safetensors checkpoint under {model_path}")


def _read_checkpoint(model_path: Path) -> dict[str, Any]:
    """Checkpoint -> ``{torch_key: jnp array}``, torch-free.

    Use the **flax** safetensors backend: it returns jnp arrays and round-trips
    bfloat16, which ``safetensors.numpy`` cannot represent at all. ``safetensors.torch``
    would read the same bytes but is a torch import, which this repo may not make.
    """
    from safetensors.flax import load_file

    state: dict[str, Any] = {}
    for path in _checkpoint_files(model_path):
        state.update(load_file(str(path)))
    return state


def _jax_path(torch_key: str) -> tuple[str, ...]:
    """HF Qwen3 ``state_dict`` name -> this module tree's parameter path.

    Torch needs no such map: its module tree is named exactly like the checkpoint.
    Flax's is not — ``nn.Dense`` calls its weight ``kernel`` and stores it transposed,
    ``nn.Embed`` calls it ``embedding``, RMSNorm calls it ``scale``, and a stack of
    layers is ``layers_0`` rather than ``layers.0`` — so the correspondence is spelled
    out here. Qwen3 keeps q/k/v separate — the extra per-head
    ``q_norm``/``k_norm`` are RMSNorms like any other — so nothing has to be split.
    """
    parts = torch_key.split(".")
    if parts[:2] == ["model", "layers"]:
        parts = ["model", f"layers_{parts[2]}", *parts[3:]]
    if parts[-1] == "bias":
        return tuple(parts)
    module = parts[-2]
    if module == "embed_tokens":
        return (*parts[:-1], "embedding")
    if module.endswith("norm"):
        return (*parts[:-1], "scale")
    return (*parts[:-1], "kernel")


def _insert(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def convert_checkpoint(model_name_or_path: str | Path, config, *, dtype=None) -> dict[str, Any]:
    """Hugging Face checkpoint -> the flax parameter pytree.

    The JAX analogue of torch's ``model.load_state_dict(_load_state_dict(model_path))``.
    Leaves keep the checkpoint's own storage dtype (Qwen3 ships bfloat16) unless
    ``dtype`` asks for a cast; ``config.dtype`` is the *compute* dtype and is applied by
    the modules. Deterministic and offline: nothing is downloaded, nothing is random.
    """
    model_path = _resolve_model_path(model_name_or_path)
    tree: dict[str, Any] = {}
    for torch_key, tensor in _read_checkpoint(model_path).items():
        path = _jax_path(torch_key)
        if path[-1] == "kernel":
            tensor = tensor.T  # torch nn.Linear [out, in] -> flax nn.Dense [in, out]
        _insert(tree, path, tensor if dtype is None else tensor.astype(dtype))
    return tree


def _special_token_str(value) -> str | None:
    """Special tokens in tokenizer_config.json may be a str or an ``AddedToken`` dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return str(value)


class _FastTokenizerAdapter:
    """Torch-free HF-compatible facade over a Rust ``tokenizers.Tokenizer``.

    JAX adjustment: torch's ``load_tokenizer`` returns an HF ``PreTrainedTokenizer``;
    the data pipeline (``Template``/``PairwiseDatasetProcessor``) uses only a small slice
    of that API — ``encode(text, add_special_tokens=…) -> list[int]``, the ``*_token`` /
    ``*_token_id`` accessors, ``add_special_tokens({...})`` and ``padding_side``. This
    adapter exposes exactly that over the Rust fast tokenizer (which reads the same
    ``tokenizer.json``, so the ids match HF exactly). It deliberately has **no**
    ``apply_chat_template`` so ``Template`` takes the fallback ``encode`` path — the same
    path HF takes for Qwen3 (which ships no chat template)."""

    def __init__(self, tok, special_tokens: dict[str, Any]):
        self._tok = tok
        self.padding_side = "right"
        self.bos_token = _special_token_str(special_tokens.get("bos_token"))
        self.eos_token = _special_token_str(special_tokens.get("eos_token"))
        self.pad_token = _special_token_str(special_tokens.get("pad_token"))
        self.unk_token = _special_token_str(special_tokens.get("unk_token"))

    def _id(self, token: str | None):
        return self._tok.token_to_id(token) if token is not None else None

    @property
    def pad_token_id(self):
        return self._id(self.pad_token)

    @property
    def eos_token_id(self):
        return self._id(self.eos_token)

    @property
    def bos_token_id(self):
        return self._id(self.bos_token)

    @property
    def unk_token_id(self):
        return self._id(self.unk_token)

    def encode(self, text, add_special_tokens: bool = True) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def add_special_tokens(self, special_tokens_dict: dict[str, Any]) -> int:
        """HF-like: register special tokens (e.g. a pad token). Sets the attribute; adds
        to the Rust tokenizer if not already present."""
        added = 0
        for key, value in special_tokens_dict.items():
            token = _special_token_str(value)
            setattr(self, key, token)
            if token is not None and self._tok.token_to_id(token) is None:
                try:
                    added += self._tok.add_special_tokens([token])
                except Exception:
                    pass
        return added


def load_tokenizer(model_args) -> TokenizerModule:
    """Load the tokenizer (mirror of torch's ``load_tokenizer``).

    JAX adjustment: returns an HF-compatible adapter over the Rust ``tokenizers`` fast
    tokenizer (torch-free) — the data pipeline uses the same ``encode``/special-token
    API and gets the same ids as torch's HF tokenizer.
    """
    tokenizer = None
    try:
        from tokenizers import Tokenizer

        model_path = _resolve_model_path(model_args.model_name_or_path)
        tok_file = model_path / "tokenizer.json"
        if tok_file.is_file():
            raw = Tokenizer.from_file(str(tok_file))
            special: dict[str, Any] = {}
            cfg_file = model_path / "tokenizer_config.json"
            if cfg_file.is_file():
                cfg = json.loads(cfg_file.read_text())
                special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
            tokenizer = _FastTokenizerAdapter(raw, special)
    except Exception:
        tokenizer = None
    return {"tokenizer": tokenizer, "processor": None}


# --------------------------------------------------------------------------- LoRA
# JAX mirror of torch_orig's PEFT LoRA (``model/loader.py``: ``_lora_exclude_pattern`` +
# ``_apply_lora`` -> ``get_peft_model``). torch targets ``"all-linear"``, i.e. every
# ``nn.Linear`` in the model minus ``exclude_modules``; spelled out for flax that is the
# seven linear projections of the LM trunk, below. ``lm_head`` is excluded by torch itself
# (``exclude_lm_head=target_modules == "all-linear"``); Qwen3 keeps it untied, so it is a
# real kernel here and ``lora_exclude_markers`` is what drops it. The per-head
# ``q_norm``/``k_norm`` are RMSNorms, not linears, so PEFT never touches them. PPO's
# ``v_head`` is not in the checkpoint -- the trainer creates it zero-initialised and trains
# it in full -- so it is absent here too.
_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: Mirror of torch's ``_lora_exclude_pattern`` marker list. ``freeze_vision_tower`` and
#: ``freeze_multi_modal_projector`` both default True, so those subtrees get no adapter.
#: Vacuous for a text-only family, but the loaders are one mirror and this is the set
#: torch builds its exclusion regex from.
_VISION_MARKERS = ("vision_tower", "vision_model", "visual", "vision_encoder", "image_tower")
_PROJECTOR_MARKERS = ("multi_modal_projector", "projector", "mm_projector", "merger")


def is_lora(args) -> bool:
    """``finetuning_type == "lora"`` — the switch torch's ``_apply_lora`` reads."""
    return str(getattr(args, "finetuning_type", "full")) == "lora"


def lora_exclude_markers(finetuning_args, exclude_lm_head: bool = True) -> tuple[str, ...]:
    """Path substrings that must not receive an adapter (torch ``_lora_exclude_pattern``)."""
    markers: list[str] = []
    if getattr(finetuning_args, "freeze_vision_tower", True):
        markers.extend(_VISION_MARKERS)
    if getattr(finetuning_args, "freeze_multi_modal_projector", True):
        markers.extend(_PROJECTOR_MARKERS)
    if exclude_lm_head:
        markers.append("lm_head")
    return tuple(markers)


def _path_str(path) -> str:
    return "/".join(str(p.key if hasattr(p, "key") else p) for p in path)


def is_adapter_path(path, extra=()) -> bool:
    """True for the leaves LoRA trains: the adapters, plus anything named in ``extra``
    (PPO passes ``("v_head",)`` — the value head is not part of the frozen checkpoint)."""
    ps = _path_str(path)
    if "_lora_A" in ps or "_lora_B" in ps:
        return True
    return any(e in ps for e in extra)


def add_adapters(params, rank: int, dtype, key=None, exclude=(), restrict=None):
    """Insert ``<name>_lora_A`` / ``<name>_lora_B`` beside every targeted ``<name>/kernel``.

    Sibling leaves rather than a wrapper module, so the base kernels keep their converted
    paths and ``transformers_impl``'s ``_proj`` can pick the pair up by name in-forward.
    ``A`` gets PEFT's ``kaiming_uniform_(a=√5)`` (see ``lora_a_init``) and ``B`` is zeros,
    so the adapter is a no-op at step 0 exactly as in torch.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    rank = int(rank)

    def walk(node, prefix, k):
        if not isinstance(node, dict):
            return node, k
        out = {}
        for name, child in node.items():
            out[name], k = walk(child, f"{prefix}/{name}", k)
        for name in list(node.keys()):
            child = node[name]
            path = f"{prefix}/{name}"
            if any(marker in path for marker in exclude):
                continue
            if (name in _TARGET_NAMES and isinstance(child, dict) and "kernel" in child
                    and (restrict is None or restrict in prefix)):
                in_d, out_d = child["kernel"].shape
                k, ka = jax.random.split(k)
                out[f"{name}_lora_A"] = lora_a_init(in_d)(ka, (in_d, rank), dtype)
                out[f"{name}_lora_B"] = jnp.zeros((rank, out_d), dtype)
        return out, k

    merged, _ = walk(params, "", key)
    return merged


def has_adapters(params) -> bool:
    """True when ``load_model`` put LoRA adapters in this tree.

    The switch the trainers read, rather than ``is_lora(args)``: it is a property of the
    params actually in hand, and unlike ``bool(split(...)[1])`` it is not confused by PPO's
    ``v_head``, which is trainable under full finetuning too.
    """
    return any("_lora_A" in _path_str(path)
               for path, _ in jax.tree_util.tree_leaves_with_path(params))


def split(params, extra=()):
    """``(params, {flat_name: leaf})`` — the adapter leaves, keyed for ``combine``.

    The base tree is handed back whole and closed over as a constant, so
    ``value_and_grad`` over the second element never allocates a cotangent for the frozen
    checkpoint. That is not just a speed win: on the 7B instances the full-tree gradient
    buffer is what does not fit the single GPU the verifier gets.
    """
    trainable: dict = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if is_adapter_path(path, extra):
            trainable[_path_str(path)] = leaf
    return params, trainable


def combine(frozen_base, trainable, extra=()):
    """Inverse of :func:`split` — graft the trained adapters back onto the frozen base."""

    def _f(path, leaf):
        if is_adapter_path(path, extra):
            return trainable[_path_str(path)]
        return leaf

    return jax.tree_util.tree_map_with_path(_f, frozen_base)


def load_model(tokenizer, model_args, finetuning_args, is_trainable: bool = True):
    """Instantiate the flax model and load the checkpoint's weights.

    Mirror of torch's ``load_model`` (resolve config + architecture, instantiate,
    load weights). JAX adjustments: returns ``(module, params, frozen_prefixes)`` rather
    than a stateful module, and the weights are remapped onto the flax pytree by
    ``convert_checkpoint`` — torch's ``from_pretrained`` does the equivalent implicitly,
    because its module names already match the checkpoint's. ``config`` comes from the
    checkpoint's ``config.json`` plus ``model_args.compute_dtype``. ``is_trainable`` is
    advisory (full finetuning; reference-model gradient-freezing is done by the trainer
    via ``stop_gradient``).
    """
    model_path = _resolve_model_path(model_args.model_name_or_path)
    config = config_from_pretrained(model_path, getattr(model_args, "compute_dtype", "bfloat16"))
    # Store the base weights at the compute dtype, mirroring torch_orig's load_model
    # (``model.to(dtype=compute_dtype)``): torch upcasts the checkpoint's stored dtype
    # to the compute dtype, so the JAX base must too. Otherwise the shipped forward runs
    # in the param storage dtype (the flax modules compute in it) and diverges from an
    # fp32 golden. When compute_dtype equals the checkpoint's storage dtype this is a no-op.
    params = convert_checkpoint(model_path, config, dtype=config.dtype)
    # LoRA (mirror of torch_orig's ``_apply_lora`` + ``_freeze_multimodal_modules``). torch
    # wraps the model in ``get_peft_model``, which freezes the base and leaves only the
    # adapters requires_grad; here the adapters are added to the pytree and the frozen part
    # is named back to the caller, because a flax pytree carries no requires_grad flag.
    frozen: list = []
    if is_lora(finetuning_args):
        rank = int(getattr(finetuning_args, "lora_rank", 8) or 8)
        alpha = int(getattr(finetuning_args, "lora_alpha", None) or rank * 2)
        config = _dc.replace(config, lora_rank=rank, lora_alpha=alpha)
        params = add_adapters(params, rank, config.dtype,
                              exclude=lora_exclude_markers(finetuning_args))
        frozen = ["<base>"]  # everything except *_lora_A/_lora_B is frozen
    module = Qwen3ForCausalLM(config)
    # Stash params so ``module.generate`` can run without re-threading them.
    object.__setattr__(module, "_gen_params", params)
    return module, params, frozen


def load_ref_model(tokenizer, model_args, finetuning_args, dpo_args):
    """Frozen reference policy for DPO — mirror of torch's ``load_ref_model``.

    Returns ``None`` when the loss is reference-free (ORPO/SimPO); otherwise loads the
    same checkpoint as the policy (``is_trainable=False``). The trainer takes the
    reference log-probs under ``jax.lax.stop_gradient``.
    """
    if str(getattr(dpo_args, "pref_loss", "sigmoid")) in ("orpo", "simpo"):
        return None
    if not getattr(dpo_args, "use_ref_model", True):
        return None
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
