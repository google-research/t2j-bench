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

"""Main entry point for offline batch serving.

The de-vendored stand-in for vLLM's `LLM` facade: one call that goes from a
config dict to a results file, wiring resolve_model_dir -> load_tokenizer ->
ModelConfig.from_hf -> build -> load_weights -> Engine.

TWO MODES, OFF CONFIG FLAGS, exactly as train.py carries do_train/do_eval in
the training class:
  do_generate  free-running greedy decode, num_decode_steps per prompt
  do_score     teacher-forced scoring of a supplied continuation

`do_score` is a real serving operation -- it is what vLLM exposes as
prompt_logprobs, and how perplexity and reranking are served -- not a grading
affordance. It is also what L6 grades, because teacher forcing is well-defined
even where a candidate would have decoded a different token; a free-running
comparison stops measuring precision the moment the two sequences diverge.

TEXT AND IMAGES, AND THE IMAGE ARRIVES AS A PATH
------------------------------------------------
This is the class's first multimodal family, so `ServeConfig` carries an
`images` list beside `prompts` / `prompt_token_ids`, and :func:`_prompt_inputs`
-- which replaces the text-only `_prompt_ids` every other family ships -- opens
each named file and runs it through this repo's OWN `image_processing.preprocess`
before either serving mode sees it.

THE PATH, NOT PRE-COMPUTED PIXELS, and the choice is the same one the text side
already made. `ServeConfig.prompts` carries raw STRINGS and this function
tokenizes them INSIDE the reference tree, with `prompt_token_ids` as the escape
hatch for a caller that wants to skip that step. A config that arrived carrying
`pixel_values` would be the image analogue of a `ServeConfig` that accepted only
`prompt_token_ids`: there would be nothing for the renamed function to compute,
and "this port processes an image" would be a claim about whoever built the
config. `PIL` is on `check_inference_structure.ARCH_ALLOWED_IMPORTS` for BOTH
reference trees for exactly this reason -- see that file's own note.

THE ENGINE NEVER LEARNS THE KEYWORD NAMES. `_prompt_inputs` returns one dict per
request, `input_ids` plus whatever modality arrays that request needs, and
:func:`_model_kwargs` forwards everything that is not `input_ids` to
`Engine.generate` / `Engine.prefill` UNINSPECTED. That mirrors the engine's own
`**model_kwargs` opacity (engine.py::prefill) one level up: a second modality
would be a change to the builder and to nothing else here.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from config import ModelConfig, RunConfig
from engine import Engine
from image_processing import ImageProcessorConfig, preprocess
from model.loader import load_tokenizer, load_weights, resolve_model_dir
from modeling import Qwen2_5_VLForConditionalGeneration
from sample.sampler import SamplingParams

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

#: RunConfig's own fields, so ServeConfig can hand them over without naming
#: them twice. Read off the dataclass rather than restated, because a RunConfig
#: field added later must not silently stop being forwarded.
#:
#: EVERY FAN-OUT OVER THIS TUPLE IS INTERSECTED WITH WHAT ServeConfig ACTUALLY
#: DECLARES, and that guard is load-bearing rather than defensive -- see
#: `ServeConfig.run_config` for the measurement that put it there. Deriving the
#: NAMES from RunConfig is right; assuming ServeConfig owns all of them is not.
_RUN_FIELDS = tuple(f.name for f in dataclasses.fields(RunConfig))

#: The key every input record carries. Everything else in the record is a
#: modality keyword and is forwarded to the model without being named here --
#: see :func:`_model_kwargs`.
_IDS_KEY = "input_ids"


@dataclass
class ServeConfig:
    model_name_or_path: str
    # RunConfig surface.
    page_size: int = 8
    num_pages: int = 8
    max_num_seqs: int = 1
    pages_per_seq: int = 4
    max_model_len: int = 32
    dtype: str = "bfloat16"
    seed: int = 42
    num_decode_steps: int = 8
    # The cache dtype is SEPARATE from the compute dtype. The fp32 leg pins it
    # to meta['fp32_kv_cache_dtype']; running an fp32 model against a bf16
    # serving cache loosens the achievable logprob bound by 180x while every
    # token still agrees, so it must never default silently to the compute one
    # at the call site -- it defaults HERE, visibly, in get_serve_args.
    kv_cache_dtype: str | None = None
    # Workload. Exactly one prompt form.
    prompts: list[str] | None = None
    prompt_token_ids: list[list[int]] | None = None
    # ONE IMAGE PATH PER PROMPT, `None` for a prompt that has no image. A file
    # PATH and not decoded pixels: see the module docstring. The list is
    # positional -- `images[i]` belongs to prompt `i` -- and it is 1:1 rather
    # than a dict keyed by index because that is the shape `continuations`
    # already uses for the other per-request payload, and two per-request lists
    # that pair up differently is the sort of asymmetry a caller gets wrong once
    # and never notices.
    #
    # A MULTIMODAL PROMPT'S PLACEHOLDERS ARE THE CALLER'S TO WRITE. This field
    # supplies the pixels; the `<|vision_start|> <|image_pad|>*N <|vision_end|>`
    # run in the ids is what says WHERE they go, and `modeling.py` refuses the
    # pair loudly (by placeholder count, and again by grid arithmetic) when the
    # two disagree. Expanding the placeholders here would mean this file owned
    # the merge arithmetic that `get_mrope_input_positions` and
    # `get_input_embeddings` already own, in a second place.
    images: list[str | None] | None = None
    continuations: list[list[int]] | None = None
    sampling: dict = field(default_factory=dict)
    do_generate: bool = True
    do_score: bool = False
    output_dir: str = "."

    def run_config(self, *, dtype: str | None = None) -> RunConfig:
        """Build a RunConfig in ONE call, optionally overriding `dtype`.

        RunConfig is `@dataclass(frozen=True)` (config.py), so a post-hoc
        `run_config.dtype = ...` assignment raises FrozenInstanceError, and
        `dataclasses.replace` is a second construction that a frozen, slotted,
        or `__post_init__`-validating RunConfig should not have to survive
        twice. The override is threaded through the ONE call instead, mirroring
        tests/_t2j/candidate.py::_generation_run_config, which builds the fp32
        leg's RunConfig the same way for the same reason.

        THE FAN-OUT IS INTERSECTED WITH ServeConfig'S OWN ATTRIBUTES, and that
        is a repair rather than caution. `_RUN_FIELDS` is derived from whatever
        RunConfig this repo declares, and the graded interface contract is a
        SUPERSET policy -- a port may legally carry a RunConfig with fields this
        ServeConfig has never heard of. An unconditional `getattr(self, k)`
        couples ServeConfig to RunConfig's field SET and blows up on exactly
        that legal port. For example, a port that adds one unread
        `enable_prefix_caching: bool = False` to RunConfig would raise
        `AttributeError: 'ServeConfig' object has no attribute
        'enable_prefix_caching'` out of this line, failing two graded checks on
        a port whose numerics are untouched. That is a false negative against a
        correct implementation, which this repo has already ruled is worse than
        the hole it would guard (see `candidate._observed_cache_dtype`).

        WHAT AN UNKNOWN FIELD DOES INSTEAD, both cases deliberate: one with a
        default falls back to RunConfig's OWN default, and one declared without
        a default raises RunConfig's own TypeError naming it -- an error about
        the config, raised by the config, rather than one about ServeConfig.
        """
        fields = {k: getattr(self, k) for k in _RUN_FIELDS if hasattr(self, k)}
        if dtype is not None:
            fields["dtype"] = dtype
        return RunConfig(**fields)


def get_serve_args(config: dict | None = None) -> ServeConfig:
    """Resolve a config dict into a validated ServeConfig.

    The analogue of train.py's get_train_args(config). Every rejection here is
    a contract the grader relies on, so each raises ValueError by name rather
    than dying further down.
    """
    cfg = ServeConfig(**dict(config or {}))
    if cfg.prompts is not None and cfg.prompt_token_ids is not None:
        raise ValueError(
            "pass exactly one of `prompts` (text) or `prompt_token_ids`; "
            "both were supplied")
    if cfg.prompts is None and cfg.prompt_token_ids is None:
        raise ValueError("no workload: pass `prompts` or `prompt_token_ids`")
    if cfg.kv_cache_dtype is None:
        cfg.kv_cache_dtype = cfg.dtype
    if cfg.continuations is not None:
        cfg.do_score = True
    if cfg.do_score and cfg.continuations is None:
        raise ValueError("`do_score` needs `continuations`")
    # THE IMAGE LIST PAIRS UP 1:1 WITH THE PROMPTS, refused here rather than by
    # `zip` truncating silently. `zip(prompts, images)` over a short `images`
    # would serve the tail of the batch as text with no error at all -- the
    # tokens would be plausible, the tower would simply never run, and the only
    # symptom is that `modeling.get_mrope_input_positions` complains about a
    # placeholder run with no grid several frames down. Counted against
    # whichever prompt form was supplied, since exactly one of them is set by
    # the time this runs.
    if cfg.images is not None:
        n_prompts = len(cfg.prompts if cfg.prompts is not None
                        else cfg.prompt_token_ids)
        if len(cfg.images) != n_prompts:
            raise ValueError(
                f"{len(cfg.images)} image(s) for {n_prompts} prompt(s); they "
                "pair up 1:1. Pass None in `images` for a prompt that carries "
                "no image")
    if cfg.prompt_token_ids is not None and cfg.continuations is not None:
        if len(cfg.continuations) != len(cfg.prompt_token_ids):
            raise ValueError(
                f"{len(cfg.continuations)} continuations for "
                f"{len(cfg.prompt_token_ids)} prompts; they pair up 1:1")
        for i, (p, c) in enumerate(zip(cfg.prompt_token_ids, cfg.continuations)):
            if len(p) + len(c) > cfg.max_model_len:
                raise ValueError(
                    f"request {i} needs {len(p)} + {len(c)} = {len(p) + len(c)} "
                    f"tokens but max_model_len is {cfg.max_model_len}")
    return cfg


def _build(cfg: ServeConfig):
    """resolve -> tokenizer -> config -> model -> weights -> engine.

    NAMES, not `apply_params`/`convert_checkpoint`: those are the JAX
    candidate's surface (solution/ref_jax_repo/model/loader.py), needed there
    because nnx params are handed over as an immutable pytree. torch's
    `nn.Module` already owns its storage, so torch_orig's loader does both jobs
    in one call -- `load_weights(model, model_dir)` reads the checkpoint and
    writes it into the already-constructed model in place, exactly as
    the offline freezing step does it.
    """
    model_dir = resolve_model_dir(cfg.model_name_or_path)
    tokenizer = load_tokenizer(model_dir)
    model_config = ModelConfig.from_hf(model_dir)
    model = Qwen2_5_VLForConditionalGeneration(model_config)
    load_weights(model, model_dir)
    model.eval()
    # THE ORDER HERE IS LOAD-BEARING: build at the checkpoint's native (fp32
    # module-default) precision, THEN cast the whole model to cfg.dtype in one
    # `.to()` call, and never reuse this model across a second dtype.
    #
    # `Module.to(dtype)` casts float BUFFERS as well as parameters, and
    # RotaryEmbedding's cos/sin table is a computed buffer, not a checkpoint
    # tensor -- it is built fresh, in fp32, at model-construction time, so
    # `.to(bfloat16)` here rounds it to 8 mantissa bits along with the weights.
    # The offline freezing step carries a whole apparatus for this because it
    # reuses one
    # model across both the fp32 and native-dtype tiers, and casting back with
    # `.float()` restores the table's DTYPE but not the BITS it lost on the way
    # down -- instance.yaml's fp32 `centered_cosine_floor` row exists
    # specifically to catch a port that carries a bf16-rounded rotary table
    # into its "fp32" leg. The vision tower has its OWN rotary table
    # (`vision.Qwen2_5_VisionRotaryEmbedding`) built the same way, so the same
    # argument covers it without a second mechanism.
    #
    # serve.py needs none of that apparatus, because `_build` is called fresh
    # for every `run_serve` and the model it returns is used at exactly one
    # dtype for its whole lifetime:
    #   dtype="float32"  -> the table is built fp32 and `.to(float32)` is a
    #                        no-op, so it stays a TRUE fp32 table.
    #   dtype="bfloat16" -> `.to(bfloat16)` rounds the table exactly once,
    #                        matching the native serving tier.
    # Do NOT "optimise" this into building one model and switching its dtype
    # across calls -- that reintroduces the offline freezing step's exact defect, and every
    # token still agrees when it happens, which is what makes it dangerous.
    model.to(getattr(torch, cfg.dtype))
    return tokenizer, model, Engine(model, cfg.run_config(dtype=cfg.kv_cache_dtype))


def _prompt_inputs(cfg: ServeConfig, tokenizer) -> list[dict]:
    """One record per request: `input_ids`, plus this request's modality arrays.

    THE TEXT HALF IS `_prompt_ids` UNCHANGED -- the function every text-only
    family in this class ships, under its old name -- and on a request with no
    image the returned record has exactly one key, so nothing reaches the engine
    that did not reach it before.

    THE IMAGE HALF IS THE SAME MOVE ONE MODALITY OVER. `cfg.images[i]` is a PATH
    on disk; this opens it, hands the ENCODED bytes to this repo's own
    `image_processing.preprocess`, and returns the `(pixel_values,
    image_grid_thw)` pair the model wants. The decode and the patchification
    both happen here, inside the policed tree, for the reason the module
    docstring gives.

    THE PROCESSOR CONFIG IS READ FROM THE CHECKPOINT, ONCE, AND ONLY IF NEEDED.
    `ImageProcessorConfig.from_hf` reads `preprocessor_config.json` -- a
    SEPARATE file from the `config.json` `ModelConfig.from_hf` reads, carrying
    the geometry (`patch_size`, `merge_size`, `min_pixels`, `max_pixels`, the
    CLIP mean/std) that `smart_resize`'s two clamps are computed against. A port
    that hardcodes this checkpoint's numbers is right here and silently wrong on
    any other. `resolve_model_dir` is called again rather than threaded down
    from `_build`: it is a path join and an `exists()` check, it keeps this
    function's signature the one-config-one-tokenizer shape the text-only
    families use, and the whole branch is skipped when no request names an
    image.
    """
    ids = ([list(p) for p in cfg.prompt_token_ids]
           if cfg.prompt_token_ids is not None
           else [list(tokenizer.encode(text).ids) for text in cfg.prompts])
    if cfg.images is None or not any(cfg.images):
        return [{_IDS_KEY: row} for row in ids]

    image_config = ImageProcessorConfig.from_hf(
        resolve_model_dir(cfg.model_name_or_path))
    records = []
    for row, image in zip(ids, cfg.images):
        record: dict = {_IDS_KEY: row}
        if image is not None:
            pixel_values, grid = preprocess(Path(image).read_bytes(),
                                            image_config)
            record["pixel_values"] = pixel_values
            record["image_grid_thw"] = grid
        records.append(record)
    return records


def _model_kwargs(record: dict) -> dict:
    """Everything in an input record except the ids, forwarded UNINSPECTED.

    The engine takes `**model_kwargs` and passes them to the positions builder
    and to the model without looking at them (engine.py::prefill), precisely so
    that no shared file learns which keyword belongs to which modality. This is
    the same opacity one level up: naming `pixel_values` and `image_grid_thw`
    here would put a second copy of that knowledge in the one file that already
    has no reason to hold it, and a second modality would then have to be added
    in two places instead of one.
    """
    return {k: v for k, v in record.items() if k != _IDS_KEY}


@torch.no_grad()
def _generate_all(cfg, engine, tokenizer, inputs) -> list[dict]:
    params = SamplingParams(**cfg.sampling) if cfg.sampling else SamplingParams()
    out = []
    for i, record in enumerate(inputs):
        ids = record[_IDS_KEY]
        engine.reset()
        result = engine.generate(torch.tensor(ids, dtype=torch.int64),
                                 num_steps=cfg.num_decode_steps, params=params,
                                 **_model_kwargs(record))
        tokens = [int(t) for t in result.tokens]
        out.append({
            # THE RECORD SCHEMA IS THE CLASS'S, UNCHANGED, and specifically it
            # does NOT gain an "images" or "modalities" field. Nothing in the
            # arrays belongs in a JSON file (16x1176 floats), and a NAME echoed
            # back would prove only that the config was read, not that the tower
            # ran -- on this family that second question answers itself, because
            # a multimodal prompt served without its grid cannot silently degrade
            # to text: `modeling.get_mrope_input_positions` refuses a run of
            # `<|image_pad|>` with no grid to go with it, by name.
            "index": i,
            "prompt_token_ids": ids,
            "tokens": tokens,
            "text": tokenizer.decode(tokens),
            "step_logprobs": [float(x) for x in result.step_logprobs],
        })
    return out


@torch.no_grad()
def _score_all(cfg, engine, inputs) -> list[dict]:
    """Teacher-forced per-token logprobs of each continuation.

    THE OFF-BY-ONE IS THE WHOLE CHECK. cont[t]'s logprob is read from the
    logits at position (prompt_len + t - 1) -- the row that PREDICTS it -- so
    a port reading position (prompt_len + t) scores each token against the
    distribution that follows it. That is the mutation test_held_out_logprobs
    exists to catch, and it is invisible to every argmax check.

    THE IMAGE, IF THERE IS ONE, IS PART OF THE PREFILL AND NOT OF THE
    CONTINUATION. The whole `prompt + continuation` goes through one prefill, so
    the tower runs once, over the placeholders that live inside the PROMPT half;
    the continuation is appended after them and shifts no image position. L6's
    own set is text-only and says so in `tests/held_out_prompts.yaml`, so this
    branch is exercised by the serving check rather than by L6 -- which is a
    deliberate boundary, not an omission.
    """
    out = []
    for i, (record, cont) in enumerate(zip(inputs, cfg.continuations)):
        ids = record[_IDS_KEY]
        engine.reset()
        cont = [int(t) for t in cont]
        full = torch.tensor(ids + cont, dtype=torch.int64)
        logits = engine.prefill(full, **_model_kwargs(record)).logits
        logprobs = [
            float(torch.log_softmax(logits[len(ids) + t - 1].float(), -1)[tok])
            for t, tok in enumerate(cont)
        ]
        out.append({
            "index": i,
            "prompt_len": len(ids),
            "cont_token_ids": cont,
            "logprobs": logprobs,
            "sum_logprob": float(sum(logprobs)),
        })
    return out


def run_serve(config: dict | None = None) -> dict:
    """The whole pipeline, from a config dict to serve_results.json."""
    cfg = get_serve_args(config)
    tokenizer, _model, engine = _build(cfg)
    inputs = _prompt_inputs(cfg, tokenizer)

    results: dict = {"config": {
        "model_name_or_path": cfg.model_name_or_path,
        "dtype": cfg.dtype, "kv_cache_dtype": cfg.kv_cache_dtype,
        "num_decode_steps": cfg.num_decode_steps,
        # The SAME intersection as `ServeConfig.run_config`, for the same
        # reason: a candidate's RunConfig may legally carry fields ServeConfig
        # does not declare, and echoing the resolved config must not be the
        # thing that fails it. An unknown field is simply not echoed.
        **{k: getattr(cfg, k) for k in _RUN_FIELDS
           if k != "dtype" and hasattr(cfg, k)},
    }}
    if cfg.do_generate:
        logger.info("*** Generating for %d prompt(s) ***", len(inputs))
        results["generations"] = _generate_all(cfg, engine, tokenizer, inputs)
    if cfg.do_score:
        logger.info("*** Scoring %d continuation(s) ***", len(inputs))
        results["scores"] = _score_all(cfg, engine, inputs)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "serve_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


if __name__ == '__main__':
    run_serve()
