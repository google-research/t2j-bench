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

STACKED_PARAMS is byte-identical to torch_orig's on purpose. The post-fusion key
set is what test_param_correspondence grades, so the two sides must agree on it
by construction rather than by coincidence: q, then k, then v along dim 0 for
qkv_proj; gate, then up for gate_up_proj.

safetensors.flax, not .numpy: 11 of the baked checkpoints are bf16 and
safetensors.numpy cannot represent bf16 at all. `.torch` is unimportable in the
verifier, which is exactly the behaviour the purity argument wants.

STRICTNESS mirrors torch_orig's `load_weights`, which raises KeyError for ANY
parameter absent from the checkpoint. The sole exemption is `lm_head.weight`,
and it is conditioned on `config.tie_word_embeddings`, not on the name: an
untied model genuinely needs its own unembedding and its absence is a real
partial load. Falling through instead would hand back a model holding zeros for
that tensor -- which is what produces "the port is subtly wrong" three days
later, a long way from the cause.

FUSED BIASES ARE COVERED, and on this instance they have to be. The paragraph
that used to sit here recorded the opposite -- "the fusion table is
`.weight`-only ... a config with attention_bias=True leaves `qkv_proj.bias`
unwritten on the torch side ... unreachable for Llama-3, whose attention_bias is
False". Qwen2 reaches it: its QKV bias is architectural rather than
config-driven, so `modeling.Qwen2Attention` hardcodes bias=True and the model
genuinely has 24 `qkv_proj.bias` parameters. Both sides now fuse the bias with
the same row offsets as the weight; the torch side's `load_weights` grew the
same suffix loop. `apply_params`' both-directions coverage check is what would
have caught a one-sided fix -- a converted set missing the bias raises
"'...qkv_proj.bias' was never written" rather than loading partially.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from safetensors.flax import load_file

DATA_ROOT = Path("/data")

# (fused parameter name, HF shard name, index of the shard within the fusion).
# Order IS the layout: q, k, v along dim 0 for qkv_proj; gate, up for gate_up_proj.
STACKED_PARAMS: tuple[tuple[str, str, int], ...] = (
    ("qkv_proj", "q_proj", 0),
    ("qkv_proj", "k_proj", 1),
    ("qkv_proj", "v_proj", 2),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)

# The fused parameters that ALSO carry a bias. Architectural, not config-driven:
# `modeling.Qwen2Attention` hardcodes bias=True on qkv_proj and bias=False on
# o_proj, mirroring vLLM, because Qwen2.5's config.json has no `attention_bias`
# key to read. gate_up_proj is bias-free on both sides, so it is absent here.
#
# DECLARED RATHER THAN SNIFFED FROM THE CHECKPOINT. `convert_checkpoint` is
# handed a config and a directory, never the model, so "does this fusion have a
# bias" cannot be answered by inspecting what happens to be present without
# making a checkpoint that OMITS the bias indistinguishable from a bias-free
# layer -- which is precisely the silent partial load the rest of this file
# exists to refuse. Declaring it means a missing bias shard raises, exactly as a
# missing weight shard does.
_FUSED_WITH_BIAS: frozenset[str] = frozenset({"qkv_proj"})

# The non-fused parameters of one decoder layer, relative to `model.layers.{i}`.
_PLAIN_LAYER_PARAMS: tuple[str, ...] = (
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


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

    The returned key set is exactly the model's parameter name set: the nine
    attribute paths, with q/k/v fused into `qkv_proj.weight` and gate/up into
    `gate_up_proj.weight`, and no transpose -- every array keeps the
    checkpoint's [out, in] orientation, which is also the orientation
    `layers/linear.py` stores.
    """
    model_dir = Path(model_dir)
    checkpoint = _read_shards(model_dir)

    hd = config.head_dim
    q_size = config.num_attention_heads * hd
    kv_size = config.num_key_value_heads * hd
    inter = config.intermediate_size
    sizes = {"qkv_proj": [q_size, kv_size, kv_size], "gate_up_proj": [inter, inter]}

    out: dict = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for fused, prefix in (("qkv_proj", f"{p}.self_attn"),
                              ("gate_up_proj", f"{p}.mlp")):
            shard_sizes = sizes[fused]
            suffixes = ("weight", "bias") if fused in _FUSED_WITH_BIAS else ("weight",)
            for suffix in suffixes:
                parts = []
                # Sorted by the third field, so STACKED_PARAMS' index is what
                # determines the layout rather than the tuple's textual order --
                # the same thing torch_orig's `sum(sizes[:index])` offset does.
                # The bias is fused on the SAME axis and in the SAME order as
                # the weight: a [out] vector against an [out, in] matrix, so
                # axis 0 is the fused axis for both.
                for _, shard, index in sorted(
                    (row for row in STACKED_PARAMS if row[0] == fused),
                    key=lambda row: row[2],
                ):
                    src_key = f"{prefix}.{shard}.{suffix}"
                    if src_key not in checkpoint:
                        start = sum(shard_sizes[:index])
                        raise KeyError(
                            f"checkpoint under {model_dir} has no '{src_key}', so "
                            f"rows [{start}:{start + shard_sizes[index]}] of "
                            f"'{prefix}.{fused}.{suffix}' would stay uninitialised. A "
                            "fused parameter is only as loaded as its least-loaded "
                            "shard."
                        )
                    parts.append(checkpoint[src_key])
                out[f"{prefix}.{fused}.{suffix}"] = jnp.concatenate(parts, axis=0)
        for rest in _PLAIN_LAYER_PARAMS:
            out[f"{p}.{rest}"] = _require(checkpoint, f"{p}.{rest}", model_dir)

    out["model.embed_tokens.weight"] = _require(
        checkpoint, "model.embed_tokens.weight", model_dir)
    out["model.norm.weight"] = _require(checkpoint, "model.norm.weight", model_dir)
    # Config-conditioned, not name-conditioned: a tied checkpoint legitimately
    # omits lm_head.weight, an untied one does not.
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
    """`model.layers.0.mlp.down_proj.weight` -> the nnx.Param object itself.

    Resolved by attribute access rather than through a flattened state, so a
    tied `lm_head.weight` lands on the SAME object as
    `model.embed_tokens.weight` -- which is the whole point of the tie and the
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


def apply_params(model, params: dict) -> set[str]:
    """Write `params` into `model`'s nnx.Params in place. Returns the names written.

    In place -- `param[...] = value` -- and never by rebinding the attribute.
    On a tied checkpoint the two are NOT equivalent: `lm_head.weight` and
    `model.embed_tokens.weight` are the same nnx.Param object, so rebinding one
    would orphan the other at its zeros while every name-level coverage check
    still passed. torch_orig's `load_weights` documents the identical trap for
    `copy_` versus `.data =`.

    Coverage is enforced in both directions: an unknown name raises, and so does
    a parameter of the model that `params` never mentions.
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
