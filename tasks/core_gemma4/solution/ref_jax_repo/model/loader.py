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

STACKED_PARAMS, CHECKPOINT_RENAMES and CHECKPOINT_PREFIXES are byte-identical to
torch_orig's on purpose. The post-fusion key set is what test_param_correspondence
grades, so the two sides must agree on it by construction rather than by
coincidence: q, then k, then v along dim 0 of qkv_proj, and gate then up along
dim 0 of gate_up_proj.

THE FUSED WIDTHS ARE PER LAYER, which is the one structural difference from
`core_llama3`'s otherwise identical converter. `head_dim` is 256 on a sliding
layer and 512 on a full one, and `intermediate_size` is 6144 on an allocating
layer and 12288 on a KV-shared one, so the shard sizes are recomputed inside the
layer loop from `config.head_dim_for_layer` and
`config.intermediate_size_for_layer` rather than hoisted above it. A converter
that hoists them is correct for the first fifteen layers.

THE PREFIX IS THE ONLY NAME CHANGE, and it is a whole component rather than a
rename: the checkpoint stores the text stack under `model.language_model.*`
because it is shared with the `Gemma4ForConditionalGeneration` bundle, and this
tree is `model.*`. See torch_orig's loader for the two upstream citations.
`CHECKPOINT_RENAMES` is EMPTY -- vLLM's attribute names are the checkpoint's on
this family, unlike gpt_oss's `attn` / `self_attn` split.

safetensors.flax, not .numpy: the checkpoint is bf16 and safetensors.numpy
cannot represent bf16 at all. `.torch` is unimportable in the verifier, which is
exactly the behaviour the purity argument wants.

STRICTNESS mirrors torch_orig's `load_weights`. The sole exemption is
`lm_head.weight`, conditioned on `config.tie_word_embeddings` and NOT on the
name -- and gemma-4 IS tied, so it fires on every load.

`layer_scalar` IS IN `_PLAIN_LAYER_PARAMS` and has no `.weight` suffix, because
it is a bare `(1,)` tensor rather than a module's weight. It is the one entry
here that vLLM cannot reach through `named_parameters()` at all (it is a buffer
upstream) and the one a de-vendored loader is most likely to drop; dropping it
is a silent per-layer rescale, not an error. See modeling.py.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from safetensors.flax import load_file

DATA_ROOT = Path("/data")

# (fused parameter name, HF shard name, index of the shard within the fusion).
# Order IS the layout: q, k, v along dim 0 for qkv_proj; gate, up for
# gate_up_proj -- and gate must be shard 0 because
# `layers/activation.GeluAndMul` reads the fused output as first half / second
# half.
STACKED_PARAMS: tuple[tuple[str, str, int], ...] = (
    ("qkv_proj", "q_proj", 0),
    ("qkv_proj", "k_proj", 1),
    ("qkv_proj", "v_proj", 2),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)

#: model attribute component -> HF checkpoint component. Whole components only.
#: EMPTY for gemma-4. See torch_orig's loader.
CHECKPOINT_RENAMES: tuple[tuple[str, str], ...] = ()

#: model name prefix -> HF checkpoint name prefix. The FIRST match wins.
CHECKPOINT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.", "model.language_model."),
)

# The non-fused parameters of one decoder layer, relative to `model.layers.{i}`.
# SEVENTEEN tensors per layer in the checkpoint: these fifteen plus the two
# fused ones. 35 x 17 + the four model-level tensors + `model.norm.weight` is
# 600, which is exactly the text-tensor count of the checkpoint header.
_PLAIN_LAYER_PARAMS: tuple[str, ...] = (
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "per_layer_input_gate.weight",
    "per_layer_projection.weight",
    "post_per_layer_input_norm.weight",
    # NO `self_attn.v_norm.weight`. gemma-4's V norm has no learned scale
    # (`RMSNormNoWeight` in modeling.py) and the checkpoint ships no such
    # tensor -- so a port that gives v_norm a weight fails HERE, on a KeyError,
    # rather than silently multiplying by ones.
    #
    # `layer_scalar` has NO `.weight` suffix: it is a bare (1,) tensor.
    "layer_scalar",
)

# The model-level parameters. The last three are PLE and are 46.6 % of the
# checkpoint between them.
_MODEL_PARAMS: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.embed_tokens_per_layer.weight",
    "model.per_layer_model_projection.weight",
    "model.per_layer_projection_norm.weight",
)


def checkpoint_key(name: str) -> str:
    """The checkpoint tensor name for a model parameter name.

    Component-wise for the renames, never a substring replace; the prefix map is
    anchored at the start. See torch_orig's loader for the full argument.
    """
    renames = dict(CHECKPOINT_RENAMES)
    name = ".".join(renames.get(part, part) for part in name.split("."))
    for src, dst in CHECKPOINT_PREFIXES:
        if name.startswith(src):
            return dst + name[len(src):]
    return name


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
    fused into `qkv_proj.weight` and gate/up into `gate_up_proj.weight`, and no
    transpose -- every array keeps the checkpoint's [out, in] orientation, which
    is also what `layers/linear.py` stores.

    A PULL MODEL: it names the tensors it needs and never walks the checkpoint,
    which is what lets a TEXT-ONLY port read a tri-modal checkpoint. The 475 M
    parameters of vision and audio tower are simply never asked for. Coverage is
    enforced model -> checkpoint here and checkpoint-key -> model in
    :func:`apply_params`, never checkpoint -> model.
    """
    model_dir = Path(model_dir)
    checkpoint = _read_shards(model_dir)

    out: dict = {}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        # PER LAYER, inside the loop. head_dim is 256 or 512 and the MLP is
        # 6144 or 12288 wide; hoisting either out of this loop is correct for
        # the first fifteen layers and wrong for the rest.
        hd = config.head_dim_for_layer(i)
        q_size = config.num_attention_heads * hd
        kv_size = config.num_kv_heads_for_layer(i) * hd
        inter = config.intermediate_size_for_layer(i)
        sizes = {"qkv_proj": [q_size, kv_size, kv_size],
                 "gate_up_proj": [inter, inter]}
        for fused, prefix in (("qkv_proj", f"{p}.self_attn"),
                              ("gate_up_proj", f"{p}.mlp")):
            shard_sizes = sizes[fused]
            parts = []
            # Sorted by the third field, so STACKED_PARAMS' index is what
            # determines the layout rather than the tuple's textual order --
            # the same thing torch_orig's `sum(sizes[:index])` offset does.
            for _, shard, index in sorted(
                (row for row in STACKED_PARAMS if row[0] == fused),
                key=lambda row: row[2],
            ):
                name = f"{prefix}.{shard}.weight"
                src_key = checkpoint_key(name)
                if src_key not in checkpoint:
                    start = sum(shard_sizes[:index])
                    raise KeyError(
                        f"checkpoint under {model_dir} has no '{src_key}', so "
                        f"rows [{start}:{start + shard_sizes[index]}] of "
                        f"'{prefix}.{fused}.weight' would stay uninitialised. A "
                        "fused parameter is only as loaded as its least-loaded "
                        "shard."
                    )
                parts.append(checkpoint[src_key])
            out[f"{prefix}.{fused}.weight"] = jnp.concatenate(parts, axis=0)
        for rest in _PLAIN_LAYER_PARAMS:
            name = f"{p}.{rest}"
            out[name] = _require(checkpoint, checkpoint_key(name), model_dir)

    for name in _MODEL_PARAMS:
        out[name] = _require(checkpoint, checkpoint_key(name), model_dir)
    # Config-conditioned, not name-conditioned: a tied checkpoint legitimately
    # omits lm_head.weight, an untied one does not. gemma-4 is tied and ships
    # none, so the first arm is the one that runs.
    out["lm_head.weight"] = (
        out["model.embed_tokens.weight"] if config.tie_word_embeddings
        else _require(checkpoint, checkpoint_key("lm_head.weight"), model_dir)
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
