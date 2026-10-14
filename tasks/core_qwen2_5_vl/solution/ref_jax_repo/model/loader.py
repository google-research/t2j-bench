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

safetensors.flax, not .numpy: the baked checkpoint is bf16 and safetensors.numpy
cannot represent bf16 at all. `.torch` is unimportable in the verifier, which is
exactly the behaviour the purity argument wants.

THERE IS NO RENAME TABLE HERE, and for this family that is a RESULT rather than
an omission -- `modeling.py` keeps the checkpoint's own `model.` / `visual.`
spelling, argued in its docstring. Every name this file produces IS a checkpoint
key, verified against `model.safetensors.index.json`'s 824 entries.

WHAT THE TOWER ADDS
-------------------
  * `visual.blocks.N.mlp.gate_proj|up_proj` fuse through the SAME
    STACKED_PARAMS rule as the text MLP, biases included -- the tower's SwiGLU
    carries biases where the text stack's does not, which is why the bias flag
    is declared PER SITE below rather than per fused NAME. A single
    `_FUSED_WITH_BIAS` set keyed on `gate_up_proj` (which is what the text-only
    families ship) would be wrong in both directions at once here: it would
    either drop 64 tower biases or invent 36 text ones.
  * `visual.blocks.N.attn.qkv` is ALREADY packed in the checkpoint, so it is not
    a fusion at all -- it is a plain copy, and the module is named `qkv` (vLLM's
    name) rather than `qkv_proj` for exactly that reason. That the two spellings
    differ by precisely the suffix that selects fusion is a coincidence of
    upstream's naming, so it is stated rather than relied on silently.
  * `visual.merger.mlp.0` / `.2` are LIST indices with a GELU at 1, and
    `visual.patch_embed.proj.weight` is a rank-5 Conv3d kernel that this side
    stores at the same rank and contracts by reshaping. Both are plain copies:
    no transpose, no reshape, no reordering -- `apply_params` refuses a shape
    that does not already match, which is what makes a "helpful" converter fail
    loudly instead of quietly.

STRICTNESS mirrors torch_orig's `load_weights`, which raises KeyError for ANY
parameter absent from the checkpoint. The sole exemption is `lm_head.weight`,
and it is conditioned on `config.tie_word_embeddings`, not on the name: an
untied model genuinely needs its own unembedding and its absence is a real
partial load. Falling through instead would hand back a model holding zeros for
that tensor -- which is what produces "the port is subtly wrong" three days
later, a long way from the cause.
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

# (submodule, fused name, does the fusion carry a bias) for one DECODER layer.
#
# DECLARED RATHER THAN SNIFFED FROM THE CHECKPOINT. `convert_checkpoint` is
# handed a config and a directory, never the model, so "does this fusion have a
# bias" cannot be answered by inspecting what happens to be present without
# making a checkpoint that OMITS the bias indistinguishable from a bias-free
# layer -- which is precisely the silent partial load the rest of this file
# exists to refuse. Declaring it means a missing bias shard raises, exactly as a
# missing weight shard does.
#
# The values are architectural, not config-driven: `modeling.Qwen2Attention`
# hardcodes bias=True on qkv_proj and bias=False on o_proj, mirroring vLLM,
# because Qwen2.5's config.json has no `attention_bias` key to read.
_TEXT_FUSIONS: tuple[tuple[str, str, bool], ...] = (
    ("self_attn", "qkv_proj", True),
    ("mlp", "gate_up_proj", False),
)

# ...and for one TOWER block. Same table shape, opposite answer on the bias:
# `vision.Qwen2_5_VisionMLP` is constructed with bias=True, and the checkpoint
# ships `visual.blocks.N.mlp.{gate,up,down}_proj.bias` to match.
_VISION_FUSIONS: tuple[tuple[str, str, bool], ...] = (
    ("mlp", "gate_up_proj", True),
)

# The non-fused parameters of one decoder layer, relative to `model.layers.{i}`.
_PLAIN_LAYER_PARAMS: tuple[str, ...] = (
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

# ...and of one tower block, relative to `visual.blocks.{i}`. `attn.qkv` is here
# and not in _VISION_FUSIONS because the checkpoint already ships it packed.
_PLAIN_BLOCK_PARAMS: tuple[str, ...] = (
    "norm1.weight",
    "norm2.weight",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "mlp.down_proj.weight",
    "mlp.down_proj.bias",
)

# ...and the tower's block-independent parameters, relative to `visual`.
_PLAIN_TOWER_PARAMS: tuple[str, ...] = (
    "patch_embed.proj.weight",
    "merger.ln_q.weight",
    "merger.mlp.0.weight",
    "merger.mlp.0.bias",
    "merger.mlp.2.weight",
    "merger.mlp.2.bias",
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


def _fuse(checkpoint: dict, model_dir: Path, prefix: str, fused: str,
          shard_sizes: list[int], *, bias: bool, out: dict) -> None:
    """Write `{prefix}.{fused}.weight` (and `.bias`) into `out`, concatenated.

    Sorted by STACKED_PARAMS' third field, so the tuple's index -- not its
    textual order -- determines the layout, which is the same thing
    torch_orig's `sum(sizes[:index])` offset does. The bias is fused on the SAME
    axis and in the SAME order as the weight: a [out] vector against an
    [out, in] matrix, so axis 0 is the fused axis for both.
    """
    for suffix in (("weight", "bias") if bias else ("weight",)):
        parts = []
        for _, shard, index in sorted(
            (row for row in STACKED_PARAMS if row[0] == fused),
            key=lambda row: row[2],
        ):
            src_key = f"{prefix}.{shard}.{suffix}"
            if src_key not in checkpoint:
                start = sum(shard_sizes[:index])
                raise KeyError(
                    f"checkpoint under {model_dir} has no '{src_key}', so rows "
                    f"[{start}:{start + shard_sizes[index]}] of "
                    f"'{prefix}.{fused}.{suffix}' would stay uninitialised. A "
                    "fused parameter is only as loaded as its least-loaded "
                    "shard."
                )
            parts.append(checkpoint[src_key])
        out[f"{prefix}.{fused}.{suffix}"] = jnp.concatenate(parts, axis=0)


def convert_checkpoint(model_dir: str | Path, config) -> dict:
    """Return {post-fusion parameter name: jnp array}, or raise.

    The returned key set is exactly the model's parameter name set: the decoder's
    nine attribute paths with q/k/v fused into `qkv_proj` and gate/up into
    `gate_up_proj`, plus the tower's ten per-block paths and its six
    block-independent ones -- and no transpose. Every array keeps the
    checkpoint's own orientation, which is also what `layers/linear.py` and
    `vision.py` store.
    """
    model_dir = Path(model_dir)
    checkpoint = _read_shards(model_dir)

    hd = config.head_dim
    q_size = config.num_attention_heads * hd
    kv_size = config.num_key_value_heads * hd
    inter = config.intermediate_size
    text_sizes = {"qkv_proj": [q_size, kv_size, kv_size],
                  "gate_up_proj": [inter, inter]}

    out: dict = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for submodule, fused, bias in _TEXT_FUSIONS:
            _fuse(checkpoint, model_dir, f"{p}.{submodule}", fused,
                  text_sizes[fused], bias=bias, out=out)
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

    # -- the tower ---------------------------------------------------------- #
    vision = config.vision_config
    vision_sizes = {"gate_up_proj": [vision.intermediate_size,
                                     vision.intermediate_size]}
    for i in range(vision.depth):
        p = f"visual.blocks.{i}"
        for submodule, fused, bias in _VISION_FUSIONS:
            _fuse(checkpoint, model_dir, f"{p}.{submodule}", fused,
                  vision_sizes[fused], bias=bias, out=out)
        for rest in _PLAIN_BLOCK_PARAMS:
            out[f"{p}.{rest}"] = _require(checkpoint, f"{p}.{rest}", model_dir)
    for rest in _PLAIN_TOWER_PARAMS:
        out[f"visual.{rest}"] = _require(checkpoint, f"visual.{rest}", model_dir)
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
    thing a state-keyed walk cannot see, because flax dedups it away. The digit
    branch also carries `visual.merger.mlp.0` / `.2`, which are list indices
    rather than attributes on this family.
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
                "[out, in] orientation on both sides and the patch embedding "
                "keeps the checkpoint's rank-5 kernel; a transposed or "
                "flattened converter shows up here."
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
