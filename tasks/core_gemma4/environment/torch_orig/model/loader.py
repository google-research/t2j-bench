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

2. FUSION. vLLM stores q/k/v as ONE qkv_proj and gate/up as ONE gate_up_proj.
   The HF checkpoint stores all five separately. STACKED_PARAMS is the mapping
   and the post-fusion key set is what test_param_correspondence grades.

   THE SHARD SIZES ARE PER LAYER, not per model, and that is what this family
   adds to `core_llama3`'s otherwise identical table. gemma-4's `head_dim` is
   256 on a sliding layer and 512 on a full one, and its `intermediate_size` is
   6144 on an allocating layer and 12288 on a KV-shared one, so `qkv_proj` is
   [2560, 1536] on 28 layers and [5120, 1536] on 7, while `gate_up_proj` is
   [12288, 1536] on 15 and [24576, 1536] on 20. `_shard_layout` reads the sizes
   off THE MODULES rather than off the config for exactly this reason -- a
   layout derived from `config.head_dim` would be right for 28 layers and
   silently truncate 7.

   NO BIASES. `attention_bias` is false and the checkpoint ships none, so the
   table is `.weight`-only and `core_gpt_oss`'s fourth `suffix` field is absent
   -- the shape `core_llama3` has.

3. THE PREFIX, AND IT IS THE ONLY NAME CHANGE. Every published gemma-4
   checkpoint is a `Gemma4ForConditionalGeneration` bundle, so the text decoder
   is stored under `model.language_model.*` beside `model.vision_tower.*` and
   `model.audio_tower.*`. This tree is the TEXT PATH ONLY and its modules are
   `model.*`, so `checkpoint_key` inserts the missing component. vLLM does the
   identical strip in the other direction (`gemma4.py:1605`: *"Checkpoint weight
   names use 'language_model.' prefix (from the Gemma4ForConditionalGeneration
   wrapper). Strip it to map to our model tree which is just 'model.*'"*), and
   tpu-inference spells it as a `WeightsMapper(orig_to_new_prefix={"model.":
   ""})`.

   THERE IS NO COMPONENT RENAME AT ALL, unlike `core_gpt_oss`. vLLM's
   `Gemma4DecoderLayer` calls the attention block `self_attn` and so does the
   checkpoint, so :data:`CHECKPOINT_RENAMES` is empty. It is kept as an empty
   table rather than deleted because `checkpoint_key` is the class's idiom for
   "the checkpoint's name for a model parameter" and a family with no renames
   should say so, not omit the concept.

WHAT IS DELIBERATELY NOT READ. 475 M parameters of `vision_tower`,
`audio_tower`, `embed_vision` and `embed_audio` -- 9.2 % of the file. Coverage
here is enforced model -> checkpoint and checkpoint-key -> model, never
checkpoint -> model, so an unread tensor is not a failure. It is a SCOPE
decision and `tests/instance_context.md` states it; this comment exists so that
a reader who counts 600 text tensors against 2011 in the file does not go
looking for a bug.

TIED, AND THIS ARM REALLY FIRES. `tie_word_embeddings: true` and the checkpoint
ships no `lm_head.weight` -- confirmed by census. `core_gpt_oss` keeps the same
arm for a checkpoint that never takes it.

`layer_scalar` IS A PARAMETER HERE AND A BUFFER UPSTREAM, so it comes through
the plain-parameter loop below like any other weight. vLLM has to reach for it
specially (`gemma4.py:1400`: `params_dict.update(dict(self.named_buffers()))
# Include buffers (e.g. layer_scalar) so they can be loaded too`), which is
exactly the sort of special case a de-vendored loader forgets -- and forgetting
it here is a silent 0.017x-to-0.87x rescale of every layer. See modeling.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

DATA_ROOT = Path("/data")

# (fused parameter name, HF shard name, index of the shard within the fusion).
# Order IS the layout: q, then k, then v along dim 0 of qkv_proj, and gate then
# up along dim 0 of gate_up_proj.
#
# THE GATE/UP ORDER IS WHAT `layers/activation.GeluAndMul` READS AS HALVES.
# gate is `x[..., :d]` there, so gate must be shard 0 here. `core_gpt_oss` is
# the counter-example that makes this worth stating: its checkpoint ships the
# two INTERLEAVED and its activation splits even/odd.
STACKED_PARAMS: tuple[tuple[str, str, int], ...] = (
    ("qkv_proj", "q_proj", 0),
    ("qkv_proj", "k_proj", 1),
    ("qkv_proj", "v_proj", 2),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)

#: model attribute component -> HF checkpoint component. Whole components only.
#: EMPTY for gemma-4: vLLM's own attribute names are the checkpoint's. See the
#: module docstring.
CHECKPOINT_RENAMES: tuple[tuple[str, str], ...] = ()

#: model name prefix -> HF checkpoint name prefix. The FIRST match wins and the
#: replacement is not re-scanned, so this is a substitution and not a rewrite
#: system. One entry, and it is the whole difference between this tree's names
#: and the checkpoint's.
CHECKPOINT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.", "model.language_model."),
)


def checkpoint_key(name: str) -> str:
    """The checkpoint tensor name for a model parameter name.

    Component-wise for the renames, never a substring replace: `'attn' in name`
    would also fire on a hypothetical `attn_norm` or `cross_attn`, and a rename
    that matches too much fails as a KeyError naming a tensor nobody wrote. The
    prefix map is anchored at the start for the same reason -- `model.` appears
    inside no other component here, but a bare `.replace` would rewrite it
    wherever it did.
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


def _shard_layout(model) -> dict[str, list[int]]:
    """Fused parameter name -> the dim-0 sizes of its shards, in order.

    READ OFF THE MODULES, per layer, and not derived from the config: on this
    family neither fused width is uniform. See the module docstring, point 2.
    """
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
    lm_head.weight, which a tied checkpoint legitimately omits.

    Every write is an in-place `copy_` into the existing Parameter, never a
    rebind of `.data`. On a tied checkpoint the two are NOT equivalent:
    lm_head.weight and model.embed_tokens.weight are the same Parameter object,
    so rebinding the embedding's storage would silently orphan lm_head at its
    uninitialised values while every name-level coverage check still passed --
    and gemma-4 IS tied, so this is live rather than hypothetical.

    THE SHAPE MISMATCH THIS FAMILY WILL PRODUCE, and where it lands: `copy_`
    into a slice refuses a source of the wrong length, so a port that got the
    double-wide MLP boundary wrong fails HERE, loudly, on layer 15 -- the one
    of gemma-4's four heterogeneities that cannot become a silent wrong answer.
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
            src_key = checkpoint_key(f"{pname.replace(fused, shard_name)}.weight")
            if src_key not in checkpoint:
                raise KeyError(
                    f"checkpoint under {model_dir} has no '{src_key}', so rows "
                    f"[{sum(sizes[:index])}:{sum(sizes[:index]) + sizes[index]}] "
                    f"of '{pname}.weight' would stay uninitialised. A fused "
                    "parameter is only as loaded as its least-loaded shard."
                )
            start = sum(sizes[:index])
            dst = params[f"{pname}.weight"]
            with torch.no_grad():
                dst[start : start + sizes[index]].copy_(checkpoint[src_key])
            loaded.add(f"{pname}.weight")

    fused_names = {f for f, _, _ in STACKED_PARAMS}
    shard_names = {s for _, s, _ in STACKED_PARAMS}
    for name, param in params.items():
        if any(f in name for f in fused_names):
            continue
        src_key = checkpoint_key(name)
        if src_key in checkpoint:
            with torch.no_grad():
                param.copy_(checkpoint[src_key])
            loaded.add(name)
        elif name == "lm_head.weight" and model.config.tie_word_embeddings:
            # gemma-4 ships no lm_head.weight; it is tied to the embedding and
            # modeling.py already aliased the Parameter object, so the
            # embedding's own load already covers it. Note this arm is
            # belt-and-braces: named_parameters() dedups shared Parameters, so
            # a tied lm_head.weight is not a key of `params` in the first
            # place. Do NOT "activate" it by passing remove_duplicate=False --
            # a tied checkpoint that DOES ship lm_head.weight would then hit
            # the branch above and copy the unembedding over the embedding.
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
