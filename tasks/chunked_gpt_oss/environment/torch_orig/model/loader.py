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

"""Checkpoint loading, de-vendored from vLLM's per-model `load_weights`.

Three things here are the deliverable a candidate must reproduce, and all three
are graded:

1. RESOLUTION. model_name_or_path is resolved to /data/{repo_id} and raises
   rather than reaching the hub. Both images bake the same directory, so the
   candidate's convert_checkpoint provably reads the same bytes the reference
   did.

2. FUSION. vLLM stores q/k/v as ONE qkv_proj. The HF checkpoint stores them
   separately. STACKED_PARAMS is the mapping and the post-fusion key set is what
   test_param_correspondence grades.

   THE BIASES FUSE TOO, and that is where this family differs from
   `core_llama3`. gpt-oss sets `attention_bias: true`, so the checkpoint ships
   `q_proj.bias` [4096], `k_proj.bias` [512] and `v_proj.bias` [512] alongside
   the weights, and they concatenate into `qkv_proj.bias` [5120] in the same
   order. `core_llama3`'s table is `.weight`-only and its plain-parameter loop
   skips every name containing a fused name, so on a bias-carrying config it
   leaves `qkv_proj.bias` UNWRITTEN -- a gap its own docstring records as
   unreachable for Llama-3. It is reachable here, so the table carries a
   `suffix` field and the fusion runs over both.

   There is NO gate/up fusion. gpt-oss ships `gate_up_proj_*` already fused --
   and INTERLEAVED, gate on the even rows and up on the odd ones, which
   `layers/activation.py::ClampedSwiGLU` consumes directly.

3. THE PACKED EXPERTS. `gate_up_proj_blocks` / `_scales` are one logical
   parameter in two checkpoint tensors, and both are `uint8`. They are copied
   VERBATIM -- no dequant at load, which is the whole memory decision (see
   `layers/moe.py`) -- and the pair is treated as a unit by the coverage rule:
   the model declares both as parameters and the checkpoint supplies both under
   exactly those names, so the generic loop already handles them. Stated here
   because "it just works" is only reassuring if someone checked.

   `copy_` INTO A uint8 PARAMETER IS NOT A CAST. torch's `copy_` converts
   dtypes, so a checkpoint tensor arriving as anything but uint8 would be
   silently rounded into the code bytes; `_require_dtype` refuses that instead.

ONE RENAME, and only one. vLLM's `TransformerBlock` calls the attention block
`attn` where the HF checkpoint calls it `self_attn`, so every attention tensor
needs `.attn.` -> `.self_attn.` on the way in. :data:`CHECKPOINT_RENAMES` is that
table, applied to whole dotted components so `attn` can never match inside
another name. Keeping vLLM's attribute name and paying one rename is the right
side of the trade: `test_reference_units` grades the port on THIS file's tree's
attribute names, and renaming the module to suit the checkpoint would put the
divergence somewhere a reader cannot see it.

UNTIED. `tie_word_embeddings` is false, so `lm_head.weight` is a real tensor in
the checkpoint and the exemption arm below never fires. It is kept because the
rule is config-conditioned, not name-conditioned.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

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
CHECKPOINT_RENAMES: tuple[tuple[str, str], ...] = (
    ("attn", "self_attn"),
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


def _shard_layout(model) -> dict[str, list[int]]:
    """Fused parameter name -> the dim-0 sizes of its shards, in order."""
    layout: dict[str, list[int]] = {}
    for name, module in model.named_modules():
        if name.endswith("qkv_proj"):
            layout[name] = [module.q_size, module.kv_size, module.kv_size]
    return layout


def _require_dtype(tensor: torch.Tensor, target: torch.Tensor, name: str) -> None:
    """Refuse a uint8 destination fed anything but uint8.

    `Tensor.copy_` casts, and casting into an MXFP4 code byte is meaningless:
    the bytes are a bit pattern, not a number. Every other dtype pair here IS a
    legitimate cast (bf16 checkpoint into an fp32 module), so the check is
    narrow on purpose.
    """
    if target.dtype == torch.uint8 and tensor.dtype != torch.uint8:
        raise TypeError(
            f"'{name}' is {tensor.dtype} in the checkpoint but uint8 in the "
            "model. These are MXFP4 code bytes, not numbers -- copy_ would cast "
            "them and produce weights that are wrong in a way nothing later can "
            "attribute.")


def load_weights(model: torch.nn.Module, model_dir: str | Path) -> set[str]:
    """Load an HF checkpoint into the model, fusing q/k/v (weights AND biases).

    Returns the set of model parameter names that were written, so a caller can
    assert full coverage -- a silent partial load is the failure mode that
    produces "the port is subtly wrong" three days later.

    That guarantee is ENFORCED here, not merely described: any parameter with no
    corresponding checkpoint tensor raises KeyError naming it, for both the
    fused shards and the plain ones. The sole exemption is a tied
    lm_head.weight, which a tied checkpoint legitimately omits -- and which
    gpt-oss, being untied, never takes.

    Every write is an in-place `copy_` into the existing Parameter, never a
    rebind of `.data`, for the reason `core_llama3` records: on a tied
    checkpoint the two are not equivalent.
    """
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    checkpoint: dict[str, torch.Tensor] = {}
    for shard in shards:
        checkpoint.update(load_file(str(shard)))

    params = dict(model.named_parameters())
    layout = _shard_layout(model)
    loaded: set[str] = set()

    for fused, shard_name, index, suffix in STACKED_PARAMS:
        for pname, sizes in layout.items():
            if not pname.endswith(fused):
                continue
            dst_key = f"{pname}.{suffix}"
            if dst_key not in params:
                # An unbiased config declares no qkv_proj.bias; the three bias
                # rows of the table are then vacuous rather than an error.
                continue
            src_key = checkpoint_key(f"{pname.replace(fused, shard_name)}.{suffix}")
            if src_key not in checkpoint:
                raise KeyError(
                    f"checkpoint under {model_dir} has no '{src_key}', so rows "
                    f"[{sum(sizes[:index])}:{sum(sizes[:index]) + sizes[index]}] "
                    f"of '{dst_key}' would stay uninitialised. A fused "
                    "parameter is only as loaded as its least-loaded shard."
                )
            start = sum(sizes[:index])
            dst = params[dst_key]
            _require_dtype(checkpoint[src_key], dst, src_key)
            with torch.no_grad():
                dst[start : start + sizes[index]].copy_(checkpoint[src_key])
            loaded.add(dst_key)

    fused_names = {f for f, _, _, _ in STACKED_PARAMS}
    shard_names = {s for _, s, _, _ in STACKED_PARAMS}
    for name, param in params.items():
        if any(f in name for f in fused_names):
            continue
        src_key = checkpoint_key(name)
        if src_key in checkpoint:
            _require_dtype(checkpoint[src_key], param, src_key)
            with torch.no_grad():
                param.copy_(checkpoint[src_key])
            loaded.add(name)
        elif name == "lm_head.weight" and model.config.tie_word_embeddings:
            # Unreachable on gpt-oss (untied), kept because the rule is
            # config-conditioned. See core_llama3's loader for the full note on
            # why this arm must not be "activated" with remove_duplicate=False.
            continue
        else:
            raise KeyError(
                f"checkpoint under {model_dir} has no '{src_key}'. Refusing to "
                "return a partially loaded model: every parameter must come "
                "from the checkpoint, and skipping one silently is what "
                "produces 'the port is subtly wrong' three days later."
            )
    assert not (shard_names & {k.split(".")[-2] for k in loaded}), \
        "an unfused shard name leaked into the loaded set"
    return loaded


def load_tokenizer(model_dir: str | Path):
    """The Rust tokenizer, or None if the checkpoint ships no tokenizer.json."""
    path = Path(model_dir) / "tokenizer.json"
    if not path.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
