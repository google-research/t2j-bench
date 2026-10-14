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

"""convert_checkpoint: HF safetensors -> the parameter pytree.

This is the deliverable EVAL_FRAMEWORK_DESIGN §4.3 describes: no converted
weights are handed over, both images carry the same /data/{repo_id}, and the
grader re-drives this function at grade time. Any parity the candidate shows is
attributable to the port, not to smuggled tensors.

STACKED_PARAMS and CHECKPOINT_RENAMES are byte-identical to torch_orig's on
purpose. The post-fusion key set is what test_param_correspondence grades, so
the two sides must agree on it by construction rather than by coincidence: q,
then k, then v along dim 0, for the WEIGHT and for the BIAS.

THE BIASES FUSE, which core_llama3's `.weight`-only table does not do. gpt-oss
sets `attention_bias: true`, so `q_proj.bias` [4096], `k_proj.bias` [512] and
`v_proj.bias` [512] concatenate into `qkv_proj.bias` [5120]. There is NO gate/up
fusion: the checkpoint ships `gate_up_proj_*` already fused, and interleaved.

THE PACKED EXPERTS ARE CARRIED THROUGH AS BYTES. `gate_up_proj_blocks` /
`_scales` are uint8 and are copied verbatim -- no dequant at load, which is the
memory decision (see `layers/moe.py`) -- and `apply_params` preserves the
parameter's own dtype, so the codes are never cast. `_require_dtype` refuses a
float source for a uint8 destination rather than rounding a bit pattern.

safetensors.flax, not .numpy: the checkpoint is bf16 and safetensors.numpy
cannot represent bf16 at all. `.torch` is unimportable in the verifier, which is
exactly the behaviour the purity argument wants.

STRICTNESS mirrors torch_orig's `load_weights`. The sole exemption is
`lm_head.weight`, conditioned on `config.tie_word_embeddings` and NOT on the
name -- and gpt-oss is untied, so it never fires here.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from safetensors.flax import load_file

DATA_ROOT = Path("/data")

# (fused parameter name, HF shard name, index of the shard within the fusion,
# the tensor suffix). Order IS the layout: q, k, v along dim 0, for BOTH the
# weight and the bias.
STACKED_PARAMS: tuple[tuple[str, str, int, str], ...] = (
    ("qkv_proj", "q_proj", 0, "weight"),
    ("qkv_proj", "k_proj", 1, "weight"),
    ("qkv_proj", "v_proj", 2, "weight"),
    ("qkv_proj", "q_proj", 0, "bias"),
    ("qkv_proj", "k_proj", 1, "bias"),
    ("qkv_proj", "v_proj", 2, "bias"),
)

#: model attribute component -> HF checkpoint component. Whole components only.
#: vLLM's `TransformerBlock` calls the attention block `attn` where the HF
#: checkpoint calls it `self_attn`.
CHECKPOINT_RENAMES: tuple[tuple[str, str], ...] = (
    ("attn", "self_attn"),
)

# The non-fused parameters of one transformer block, relative to
# `model.layers.{i}`, in the MODEL's naming (`attn`, not `self_attn`).
_PLAIN_LAYER_PARAMS: tuple[str, ...] = (
    "attn.o_proj.weight",
    "attn.o_proj.bias",
    "attn.sinks",
    "mlp.router.weight",
    "mlp.router.bias",
    "mlp.experts.gate_up_proj_blocks",
    "mlp.experts.gate_up_proj_scales",
    "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj_blocks",
    "mlp.experts.down_proj_scales",
    "mlp.experts.down_proj_bias",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def checkpoint_key(name: str) -> str:
    """The checkpoint tensor name for a model parameter name.

    Component-wise, never a substring replace: `'attn' in name` would also fire
    on a hypothetical `attn_norm` or `cross_attn`, and a rename that matches too
    much fails as a KeyError naming a tensor nobody wrote.
    """
    renames = dict(CHECKPOINT_RENAMES)
    return ".".join(renames.get(part, part) for part in name.split("."))


def resolve_model_dir(model_name_or_path: str | Path) -> Path:
    """Repo id -> /data/{repo_id}. Never reaches the hub."""
    p = Path(model_name_or_path)
    candidate = p if p.is_absolute() else DATA_ROOT / p
    if not (candidate / "config.json").exists():
        raise FileNotFoundError(
            f"model '{model_name_or_path}' is not baked at {candidate}. "
            "Both images carry the checkpoint at /data/{repo_id}; grading runs "
            "with no network, so nothing can be downloaded here."
        )
    return candidate


def _read_shards(model_dir: Path) -> dict:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    checkpoint: dict = {}
    for shard in shards:
        checkpoint.update(load_file(str(shard)))
    return checkpoint


def convert_checkpoint(model_dir: str | Path, config) -> dict:
    """Return {post-fusion parameter name: jnp array}, or raise.

    The returned key set is exactly the model's parameter name set, with q/k/v
    fused into `qkv_proj.{weight,bias}` and no transpose -- every array keeps
    the checkpoint's [out, in] orientation, which is also what
    `layers/linear.py` stores. The four packed expert tensors keep their uint8
    dtype; the whole point is that nothing here materialises a dequantised
    expert.
    """
    model_dir = Path(model_dir)
    checkpoint = _read_shards(model_dir)

    hd = config.head_dim
    q_size = config.num_attention_heads * hd
    kv_size = config.num_key_value_heads * hd
    shard_sizes = [q_size, kv_size, kv_size]

    out: dict = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for suffix in ("weight", "bias"):
            if suffix == "bias" and not config.attention_bias:
                continue
            parts = []
            # Sorted by the third field, so STACKED_PARAMS' index is what
            # determines the layout rather than the tuple's textual order.
            for _, shard, index, _ in sorted(
                (row for row in STACKED_PARAMS if row[3] == suffix),
                key=lambda row: row[2],
            ):
                src_key = f"{p}.self_attn.{shard}.{suffix}"
                if src_key not in checkpoint:
                    start = sum(shard_sizes[:index])
                    raise KeyError(
                        f"checkpoint under {model_dir} has no '{src_key}', so "
                        f"rows [{start}:{start + shard_sizes[index]}] of "
                        f"'{p}.attn.qkv_proj.{suffix}' would stay "
                        "uninitialised. A fused parameter is only as loaded as "
                        "its least-loaded shard."
                    )
                parts.append(checkpoint[src_key])
            out[f"{p}.attn.qkv_proj.{suffix}"] = jnp.concatenate(parts, axis=0)
        for rest in _PLAIN_LAYER_PARAMS:
            if rest == "attn.o_proj.bias" and not config.attention_bias:
                continue
            name = f"{p}.{rest}"
            out[name] = _require(checkpoint, checkpoint_key(name), model_dir)

    out["model.embed_tokens.weight"] = _require(
        checkpoint, "model.embed_tokens.weight", model_dir)
    out["model.norm.weight"] = _require(checkpoint, "model.norm.weight", model_dir)
    # Config-conditioned, not name-conditioned: a tied checkpoint legitimately
    # omits lm_head.weight, an untied one does not. gpt-oss is untied.
    out["lm_head.weight"] = (
        out["model.embed_tokens.weight"] if config.tie_word_embeddings
        else _require(checkpoint, "lm_head.weight", model_dir)
    )
    return out


def _require(checkpoint: dict, name: str, model_dir: Path):
    if name not in checkpoint:
        raise KeyError(
            f"checkpoint under {model_dir} has no '{name}'. Refusing to return "
            "a partially converted parameter set: every parameter must come "
            "from the checkpoint, and skipping one silently is what produces "
            "'the port is subtly wrong' three days later."
        )
    return checkpoint[name]


def _resolve(model, dotted: str) -> nnx.Param:
    """`model.layers.0.mlp.router.weight` -> the nnx.Param object itself.

    Resolved by attribute access rather than through a flattened state, so a
    tied `lm_head.weight` would land on the SAME object as
    `model.embed_tokens.weight` -- which is the whole point of a tie and the
    thing a state-keyed walk cannot see, because flax dedups it away.
    """
    obj = model
    parts = dotted.split(".")
    for j, part in enumerate(parts):
        try:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        except (AttributeError, IndexError, TypeError) as exc:
            raise KeyError(
                f"'{dotted}' does not name a parameter of this model: "
                f"'{'.'.join(parts[: j + 1])}' does not exist"
            ) from exc
    if not isinstance(obj, nnx.Param):
        raise KeyError(f"'{dotted}' resolves to {type(obj).__name__}, not an nnx.Param")
    return obj


def _param_names(model) -> list[str]:
    """Every nnx.Param path in the model, deduplicated across tied references."""
    _, params, _ = nnx.split(model, nnx.Param, ...)
    return [".".join(str(part) for part in path)
            for path, _ in nnx.to_flat_state(params)]


def _require_dtype(value, param, name: str) -> None:
    """Refuse a uint8 destination fed anything but uint8.

    `astype` casts, and casting into an MXFP4 code byte is meaningless: the
    bytes are a bit pattern, not a number. Every other dtype pair here IS a
    legitimate cast (bf16 checkpoint into an fp32 parameter), so the check is
    narrow on purpose. Mirrors torch_orig's `_require_dtype`.
    """
    if param.dtype == jnp.uint8 and value.dtype != jnp.uint8:
        raise TypeError(
            f"'{name}' is {value.dtype} in the converted checkpoint but uint8 "
            "in the model. These are MXFP4 code bytes, not numbers -- a cast "
            "would produce weights that are wrong in a way nothing later can "
            "attribute.")


def apply_params(model, params: dict) -> set[str]:
    """Write `params` into `model`'s nnx.Params in place. Returns the names written.

    In place -- `param[...] = value` -- and never by rebinding the attribute, so
    a tie survives. Coverage is enforced in both directions: an unknown name
    raises, and so does a parameter of the model that `params` never mentions.
    """
    written: dict[int, str] = {}
    for name, array in params.items():
        param = _resolve(model, name)
        value = jnp.asarray(array)
        if value.shape != param.shape:
            raise ValueError(
                f"'{name}' is {value.shape} in the converted checkpoint but "
                f"{param.shape} in the model. Linear weights keep torch's "
                "[out, in] orientation on both sides; a transposed converter "
                "shows up here."
            )
        _require_dtype(value, param, name)
        param[...] = value.astype(param.dtype)
        written[id(param)] = name

    for name in _param_names(model):
        if id(_resolve(model, name)) not in written:
            raise KeyError(
                f"'{name}' was never written: the converted parameter set does "
                "not cover the model. Refusing to return a partially loaded "
                "model."
            )
    return set(params)


def load_tokenizer(model_dir: str | Path):
    """The Rust tokenizer, or None if the checkpoint ships no tokenizer.json."""
    path = Path(model_dir) / "tokenizer.json"
    if not path.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
