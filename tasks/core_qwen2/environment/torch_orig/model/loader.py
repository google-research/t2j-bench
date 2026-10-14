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

Two things here are the actual deliverable a candidate must reproduce, and both
are graded:

1. RESOLUTION. model_name_or_path is resolved to /data/{repo_id} and raises
   rather than reaching the hub. Both images bake the same directory, so the
   candidate's convert_checkpoint provably reads the same bytes the reference
   did.

2. FUSION. vLLM stores q/k/v as ONE qkv_proj and gate/up as ONE gate_up_proj.
   The HF checkpoint stores them separately. STACKED_PARAMS is the mapping, and
   the resulting post-fusion key set is what test_param_correspondence grades --
   a port that keeps three separate q/k/v tensors does not mirror vLLM and
   fails, correctly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

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
        elif name.endswith("gate_up_proj"):
            layout[name] = list(module.output_sizes)
    return layout


def load_weights(model: torch.nn.Module, model_dir: str | Path) -> set[str]:
    """Load an HF checkpoint into the model, fusing q/k/v and gate/up.

    Returns the set of model parameter names that were written, so a caller can
    assert full coverage -- a silent partial load is the failure mode that
    produces "the port is subtly wrong" three days later.

    That guarantee is ENFORCED here, not merely described: any parameter with no
    corresponding checkpoint tensor raises KeyError naming it, for both the
    fused shards and the plain ones. The sole exemption is a tied
    lm_head.weight, which a tied checkpoint legitimately omits. Returning the
    `loaded` set on top of that is belt-and-braces -- it lets a caller re-check
    the same invariant -- but the load now refuses to hand back a
    half-initialised model at all, so the invariant holds in production and not
    only under pytest.

    Every write is an in-place `copy_` into the existing Parameter, never a
    rebind of `.data`. On a tied checkpoint the two are NOT equivalent:
    lm_head.weight and model.embed_tokens.weight are the same Parameter object,
    so rebinding the embedding's storage would silently orphan lm_head at its
    uninitialised values while every name-level coverage check still passed.
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

    for fused, shard_name, index in STACKED_PARAMS:
        for pname, sizes in layout.items():
            if not pname.endswith(fused):
                continue
            # BOTH TENSORS, not just the weight. A bias-carrying family (Qwen2
            # hardcodes bias=True on qkv_proj) ships q/k/v biases unfused in the
            # checkpoint exactly as it ships the weights, and they need the same
            # row-offset fusion. Hardcoding `.weight` here would leave
            # `qkv_proj.bias` to the fallback loop below -- which skips every
            # name containing a fused name -- so it would be written by neither
            # and keep torch.empty()'s contents.
            for suffix in ("weight", "bias"):
                dst_key = f"{pname}.{suffix}"
                # A linear built with bias=False has no `.bias` parameter, and a
                # checkpoint for it carries no `.bias` tensor. Absent on BOTH
                # sides is the normal case, not an error; absent on ONE side is
                # the error, and the KeyError below still says so.
                if dst_key not in params:
                    continue
                src_key = f"{pname.replace(fused, shard_name)}.{suffix}"
                if src_key not in checkpoint:
                    raise KeyError(
                        f"checkpoint under {model_dir} has no '{src_key}', so rows "
                        f"[{sum(sizes[:index])}:{sum(sizes[:index]) + sizes[index]}] "
                        f"of '{dst_key}' would stay uninitialised. A fused "
                        "parameter is only as loaded as its least-loaded shard."
                    )
                start = sum(sizes[:index])
                dst = params[dst_key]
                with torch.no_grad():
                    dst[start : start + sizes[index]].copy_(checkpoint[src_key])
                loaded.add(dst_key)

    fused_names = {f for f, _, _ in STACKED_PARAMS}
    shard_names = {s for _, s, _ in STACKED_PARAMS}
    for name, param in params.items():
        if any(f in name for f in fused_names):
            continue
        if name in checkpoint:
            with torch.no_grad():
                param.copy_(checkpoint[name])
            loaded.add(name)
        elif name == "lm_head.weight" and model.config.tie_word_embeddings:
            # Qwen2.5-0.5B ships no lm_head.weight; it is tied to the embedding
            # and modeling.py already aliased the Parameter object, so the
            # embedding's own load already covers it. Note this arm is
            # belt-and-braces: named_parameters() dedups shared Parameters, so
            # a tied lm_head.weight is not a key of `params` in the first
            # place. Do NOT "activate" it by passing remove_duplicate=False --
            # a tied checkpoint that DOES ship lm_head.weight would then hit
            # the branch above and copy the unembedding over the embedding.
            continue
        else:
            raise KeyError(
                f"checkpoint under {model_dir} has no '{name}'. Refusing to "
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
