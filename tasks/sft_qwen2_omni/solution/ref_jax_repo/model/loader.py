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
  (``Qwen2_5OmniForConditionalGeneration`` / ``config_from_pretrained``), mirroring torch's
  ``_resolve_local_model_class`` / ``_load_local_config``.
"""

from __future__ import annotations

import dataclasses as _dc
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp

from transformers_impl.configuration import Qwen2_5OmniConfig, config_from_pretrained
from transformers_impl.modeling import Qwen2_5OmniForConditionalGeneration, lora_a_init

logger = logging.getLogger(__name__)

_LOCAL_TRANSFORMERS_PACKAGE = "transformers_impl"
_DEFAULT_CONFIG_CLASS = "Qwen2_5OmniConfig"
_DEFAULT_ARCHITECTURE = "Qwen2_5OmniForConditionalGeneration"


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


# The thinker's audio tower and the token2wav vocoder are convolutional (rank 3), and
# the vision tower's patch embedding is a Conv3d (rank 5): torch stores a conv weight
# [out, in, *spatial], flax [*spatial, in, out].
_CONV_RANKS = (3, 5)


def _jax_path(runtime_key: str, rank: int) -> tuple[str, ...]:
    """Checkpoint key -> this tree's parameter path, by leaf name and tensor rank.

    Torch needs no such map: its module tree is named exactly like the checkpoint.
    Flax's is not — ``nn.Dense`` calls its weight ``kernel`` and stores it transposed,
    ``nn.Embed`` calls it ``embedding``, and norms call their gain ``scale``. Under
    torch's one name ``weight`` it is the rank that tells them apart: 1-D is a norm
    gain, 2-D a Dense kernel unless the module is an embedding table (the audio tower's
    ``audio_bos_eos_token`` is one, despite the name), and 3-D or 5-D a convolution.
    """
    parts = runtime_key.split(".")
    if parts[-1] != "weight":
        return tuple(parts)  # biases and the tree's handful of bare tensors
    if len(parts) >= 4 and parts[-4] == "ups" and parts[-2] == "0":
        # BigVGAN's upsamplers are ConvTranspose1d, whose weight torch stores
        # [in, out, k] rather than Conv1d's [out, in, k], so the conv permutation below
        # would be wrong for them. This tree keeps them verbatim, under ``scale``.
        return (*parts[:-1], "scale")
    module = parts[-2]
    if rank == 1:
        return (*parts[:-1], "scale")
    if rank == 2:
        if "embed" in module.lower() or module == "audio_bos_eos_token":
            return (*parts[:-1], "embedding")
        return (*parts[:-1], "kernel")
    if rank in _CONV_RANKS:
        return (*parts[:-1], "kernel")
    return tuple(parts)


def _insert(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = tree
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def convert_checkpoint(model_name_or_path: str | Path, config, *, dtype=None) -> dict[str, Any]:
    """Hugging Face checkpoint -> the flax parameter pytree.

    The JAX analogue of torch's ``model.load_state_dict(_load_state_dict(model_path))``.
    Every leaf is cast to the config's compute dtype, so the tree's precision is a
    property of the run and not of the file — a real conversion here, because
    Qwen2.5-Omni-3B mixes bfloat16 weights with a float32 token2wav DiT.
    The checkpoint is already written in runtime module names, so no key rewriting
    is needed, and the thinker's ``lm_head.weight`` is its own tensor rather than a
    tie — there is nothing to add or drop.
    Deterministic and offline: nothing is downloaded, nothing is random.
    """
    model_path = _resolve_model_path(model_name_or_path)
    cast = config.dtype if dtype is None else dtype
    tree: dict[str, Any] = {}
    for file_key, tensor in _read_checkpoint(model_path).items():
        if file_key.endswith("rotary_embed.inv_freq"):
            continue  # a derived RoPE table the module recomputes, not a parameter
        path = _jax_path(file_key, tensor.ndim)
        if path[-1] == "kernel":
            # torch nn.Linear [out, in] -> flax nn.Dense [in, out]; torch conv
            # [out, in, *spatial] -> flax [*spatial, in, out].
            tensor = tensor.T if tensor.ndim == 2 else jnp.moveaxis(tensor, (0, 1), (-1, -2))
        _insert(tree, path, tensor.astype(cast))
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
    path HF takes for Qwen2 (which ships no chat template)."""

    def __init__(self, tok, special_tokens: dict[str, Any]):
        self._tok = tok
        self.padding_side = "right"
        self.bos_token = _special_token_str(special_tokens.get("bos_token"))
        self.eos_token = _special_token_str(special_tokens.get("eos_token"))
        self.pad_token = _special_token_str(special_tokens.get("pad_token"))
        self.unk_token = _special_token_str(special_tokens.get("unk_token"))

    def _id(self, token: str | None):
        return self._tok.token_to_id(token) if token is not None else None

    def convert_tokens_to_ids(self, tokens):
        """HF-like: map a token (or a list of them) to id(s); ``None`` when unknown."""
        if isinstance(tokens, (list, tuple)):
            return [self._tok.token_to_id(token) for token in tokens]
        return self._tok.token_to_id(tokens)

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


def _register_added_tokens(raw, tokenizer_config: dict) -> None:
    """Mirror HF's merge of ``added_tokens_decoder`` into the fast tokenizer.

    ``tokenizer.json`` does not always list every added token — Qwen2-VL omits
    ``<|image_pad|>`` — and a token the tokenizer does not know is split into subwords,
    which silently breaks the image-token count the model forward asserts on. HF reads
    them back from ``tokenizer_config.json``; do the same. An id that lands anywhere other
    than where the checkpoint says is an error, not something to paper over: it would
    index the wrong embedding row.
    """
    from tokenizers import AddedToken

    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    for token_id, spec in sorted(decoder.items(), key=lambda item: int(item[0])):
        spec = spec if isinstance(spec, dict) else {"content": spec}
        content = spec.get("content")
        if content is None or raw.token_to_id(content) is not None:
            continue
        token = AddedToken(
            content,
            single_word=spec.get("single_word", False),
            lstrip=spec.get("lstrip", False),
            rstrip=spec.get("rstrip", False),
            normalized=spec.get("normalized", False),
            special=spec.get("special", True),
        )
        if spec.get("special", True):
            raw.add_special_tokens([token])
        else:
            raw.add_tokens([token])
        actual = raw.token_to_id(content)
        if actual != int(token_id):
            raise ValueError(
                f"Added token {content!r} landed at id {actual}, but the checkpoint says "
                f"{token_id}; that id would index the wrong embedding row."
            )


def load_tokenizer(model_args) -> TokenizerModule:
    """Load the tokenizer + image processor (mirror of torch's ``load_tokenizer``).

    JAX adjustment: returns an HF-compatible adapter over the Rust ``tokenizers`` fast
    tokenizer (torch-free) — the data pipeline uses the same ``encode``/special-token
    API and gets the same ids as torch's HF tokenizer — paired with the vendored,
    torch-free processor from ``transformers_impl.processing``, so ``data/mm_plugin.py``
    expands the same number of image tokens and builds the same ``pixel_values``.
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
                _register_added_tokens(raw, cfg)
            tokenizer = _FastTokenizerAdapter(raw, special)
    except Exception:
        tokenizer = None
    processor = None
    try:
        from transformers_impl.processing import Qwen2_5OmniProcessor

        processor = Qwen2_5OmniProcessor.from_pretrained(
            str(_resolve_model_path(model_args.model_name_or_path)), tokenizer=tokenizer
        )
        # Mirror of torch's patch_processor: data/mm_plugin.py reads the resize bounds off
        # the processor, so ModelArguments has to be copied onto it here or they are dead.
        for attribute in ("image_max_pixels", "image_min_pixels"):
            if hasattr(model_args, attribute):
                setattr(processor, attribute, getattr(model_args, attribute))
    except Exception as error:
        # Swallowed silently, the whole image path just disappears and the run quietly
        # trains on text only; mm_plugin then raises rather than dropping the pixels.
        logger.warning("Could not load a processor for %s: %s", model_args.model_name_or_path, error)
        processor = None
    return {"tokenizer": tokenizer, "processor": processor}


# --------------------------------------------------------------------------- LoRA
# JAX mirror of torch_orig's PEFT LoRA (``model/loader.py``: ``_lora_exclude_pattern`` +
# ``_apply_lora`` -> ``get_peft_model``). torch targets ``"all-linear"``, i.e. every
# ``nn.Linear`` in the model minus ``exclude_modules``; spelled out for flax that is the
# linear projections of the thinker decoder, below -- the exact set of ``nn.Dense`` names
# ``transformers_impl.modeling`` routes through ``_proj``. ``lm_head`` is
# excluded by torch itself (``exclude_lm_head=target_modules == "all-linear"``), and the
# vision tower (``visual``) and its ``merger`` are excluded by the marker list below, as
# ``freeze_vision_tower``/``freeze_multi_modal_projector`` both default True. PPO's
# ``v_head`` is not in the checkpoint -- the trainer creates it zero-initialised and trains
# it in full -- so it is absent here too.
_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: Mirror of torch's ``_lora_exclude_pattern`` marker list. ``freeze_vision_tower`` and
#: ``freeze_multi_modal_projector`` both default True, so those subtrees get no adapter.
#: ``visual`` and ``merger`` are the two that actually bite for Qwen2.5-Omni, whose
#: converted tree carries the whole checkpoint (thinker vision/audio towers + talker +
#: token2wav), but the loaders are one mirror and this is the set torch builds its
#: exclusion regex from.
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
    module = Qwen2_5OmniForConditionalGeneration(config)
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
