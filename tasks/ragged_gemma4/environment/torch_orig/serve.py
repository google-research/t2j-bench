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
from model.loader import load_tokenizer, load_weights, resolve_model_dir
from modeling import Gemma4ForCausalLM
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


@dataclass
class ServeConfig:
    model_name_or_path: str
    # RunConfig surface.
    page_size: int = 32
    num_pages: int = 20
    max_num_seqs: int = 1
    pages_per_seq: int = 17
    max_model_len: int = 544
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
    model = Gemma4ForCausalLM(model_config)
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
    # into its "fp32" leg.
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


def _prompt_ids(cfg: ServeConfig, tokenizer) -> list[list[int]]:
    if cfg.prompt_token_ids is not None:
        return [list(p) for p in cfg.prompt_token_ids]
    return [list(tokenizer.encode(text).ids) for text in cfg.prompts]


@torch.no_grad()
def _generate_all(cfg, engine, tokenizer, prompts) -> list[dict]:
    params = SamplingParams(**cfg.sampling) if cfg.sampling else SamplingParams()
    out = []
    for i, ids in enumerate(prompts):
        engine.reset()
        result = engine.generate(torch.tensor(ids, dtype=torch.int64),
                                 num_steps=cfg.num_decode_steps, params=params)
        tokens = [int(t) for t in result.tokens]
        out.append({
            "index": i,
            "prompt_token_ids": ids,
            "tokens": tokens,
            "text": tokenizer.decode(tokens),
            "step_logprobs": [float(x) for x in result.step_logprobs],
        })
    return out


@torch.no_grad()
def _score_all(cfg, engine, prompts) -> list[dict]:
    """Teacher-forced per-token logprobs of each continuation.

    THE OFF-BY-ONE IS THE WHOLE CHECK. cont[t]'s logprob is read from the
    logits at position (prompt_len + t - 1) -- the row that PREDICTS it -- so
    a port reading position (prompt_len + t) scores each token against the
    distribution that follows it. That is the mutation test_held_out_logprobs
    exists to catch, and it is invisible to every argmax check.
    """
    out = []
    for i, (ids, cont) in enumerate(zip(prompts, cfg.continuations)):
        engine.reset()
        cont = [int(t) for t in cont]
        full = torch.tensor(ids + cont, dtype=torch.int64)
        logits = engine.prefill(full).logits
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
    prompts = _prompt_ids(cfg, tokenizer)

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
        logger.info("*** Generating for %d prompt(s) ***", len(prompts))
        results["generations"] = _generate_all(cfg, engine, tokenizer, prompts)
    if cfg.do_score:
        logger.info("*** Scoring %d continuation(s) ***", len(prompts))
        results["scores"] = _score_all(cfg, engine, prompts)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "serve_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


if __name__ == '__main__':
    run_serve()
