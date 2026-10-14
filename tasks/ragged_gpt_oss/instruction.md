# PyTorch-to-JAX Conversion: Paged-Attention Inference Engine

## Goal

Reimplement the PyTorch inference engine in `torch_orig/` as a **native JAX/Flax** repo
in `jax_repo/` with the same serving semantics. Mirror `torch_orig`'s structure and
public API -- same modules, same class and function names and signatures -- deviating
only where JAX forces it. Reimplement the GptOss model in `flax.nnx`, the paged KV cache
and the paged-attention kernel in pure JAX, keep the sampler's transforms exact, and
port the weights yourself.

Write it to **`/workspace/jax_repo/`** -- that exact absolute path, and nothing outside
it is collected. The PyTorch source you are porting is at `/workspace/torch_orig/` and
is read-only as far as the task is concerned: changing it changes nothing about how you
are graded.

The engine serves in **ragged** mode: a persistent batch of concurrent requests,
addressed by `query_start_loc` rather than by a batch dimension. One forward carries
several sequences at different stages -- some contributing their whole prompt (prefill),
some contributing a single token over KV the cache already holds (decode) -- so the
attention has to derive each query token's sequence and its absolute position from the
metadata rather than from its index. There is still no partial (chunked) prefill: a
request's prompt is consumed in one forward.

`jax_repo/` MUST NOT import `torch`, `torchvision`, `torchaudio`, `functorch` or
`torch_xla` **anywhere** -- no offline exception, not even to read the checkpoint. The
grading environment has no torch, no `transformers`, no `vllm` and no HF `datasets`, so
whatever `torch_orig` imports from those your repo has to carry itself.

## The environment you work in

Two environment variables are part of the contract, not decoration:
`JAX_DEFAULT_MATMUL_PRECISION=highest` and `--xla_gpu_autotune_level=0` in `XLA_FLAGS`.
Every graded run forces both, and the container you are working in is built with both
already set, so ordinary work inherits them. **Keep them set in anything you run
yourself.** Without the precision pin XLA lowers an fp32 matmul to TF32, and a *correct*
fp32 port then disagrees with `torch_orig` by hundreds of times the band the
full-precision comparison allows -- so the first thing an unpinned side-by-side tells
you is that your port is broken when it is not. Without the autotuner pin the same graph
can lower to different kernels in two runs of the same code, which makes a
reduced-precision comparison irreproducible run to run. If you drive `torch_orig` and
your `jax_repo` from a script, an editor task or a subprocess that starts from a clean
environment, set them there too.

### Compute and time budget

You work in a container with one visible H100 GPU, 24 CPU cores and 200 GiB of RAM, and
have up to **2 hours** of wall-clock time to produce `jax_repo/` -- enough to write, run
and debug the port. That budget is for building the port; the grading afterwards runs in
a separate offline container against artifacts frozen from a run of `torch_orig` (see
**How you are graded**).

## Instance context

This task targets one specific model, checkpoint and serving mode. The sections below
give the model identity, the checkpoint you must convert, and the shape of the paged KV
cache your engine has to maintain; everything else about the repo layout mirrors
`torch_orig`.

### Model
- `model_name_or_path`: `openai/gpt-oss-20b`
- serving mode: `ragged` -- a persistent batch of concurrent requests, addressed by `query_start_loc` rather than by a batch dimension.
- graded precision profile: `bf16_native` (dtype `bfloat16`)

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/openai/gpt-oss-20b` -- `config.json`, the tokenizer files, and the weights.
Nothing is downloaded at any point: the container has no network at grade time, this
directory is the only copy, and your `model/loader.py` reads it straight from there. The
path is a hard contract, not a default -- `resolve_model_dir` resolves
`/data/{model_name_or_path}` and the grading image bakes the same bytes the agent image
does.

**This checkpoint is 4-bit, and that changes what the fp32 tier means here.**
`openai/gpt-oss-20b` is released MXFP4-quantised: `quantization_config.quant_method` is
`mxfp4`, and 19.1 billion of its 20.9 billion parameters -- every routed expert
projection -- exist only as 4-bit codes plus one shared exponent per 32 values.
Attention, the routers, the embedding and the head are exempt (`modules_to_not_convert`)
and are bf16 like any other checkpoint. There is no fp32 export of these weights and
none is manufactured: the reference keeps the packed bytes resident and dequantises
inside the forward, which is what lets three copies of a 20.9B model coexist on one
card.

So **the fp32 band this instance is graded against is fp32 *accumulation* over 4-bit
weights**, not the both-sides-identical-fp32-weights statement the dense families in
this class make. It is still a strictly tighter gate than the bf16 band and it still
separates a correct port from an incorrect one -- what it does not do is certify
agreement at a precision the checkpoint does not have. Note that the dequantisation
itself loses nothing at either dtype: every FP4 value needs three mantissa bits and the
scale is a power of two, so the decode is exact in bfloat16 as well as in float32 and
the whole difference between the two tiers is in the accumulation.

**Declare that root as a literal.** `model/loader.py` must carry `DATA_ROOT =
Path("/data")` -- a module-level assignment of a string literal, optionally wrapped in
`Path(...)`. It is read out of your SOURCE and not at run time, because the harness
rebinds the live value on hosts that have no `/data`, so a value computed from the
environment, from a join or from another constant states nothing about the contract and
does not satisfy it. Resolve under the constant however you like; declare it literally.

**And `resolve_model_dir` must REFUSE a checkpoint that is not there.** Resolving the
path is not enough: when the directory it resolves to holds no checkpoint -- no
`config.json` under it -- the function has to RAISE rather than hand the path back.
Grading has no network, so a path nothing checked names a directory whose weights do not
exist, and the failure then surfaces hundreds of lines away inside `convert_checkpoint`
instead of at the lookup that came up empty. A bare `return DATA_ROOT /
model_name_or_path` does not satisfy this. Raise something that reads as an absent
checkpoint -- the reference raises `FileNotFoundError`, and a `ValueError` or `OSError`
is equally acceptable -- and make sure it is a deliberate refusal rather than an
incidental crash: an `AttributeError`, a `TypeError` or an `ImportError` out of that
function is the shape of a bug in your loader, not of a lookup that found nothing, and
does not count as refusing.

### The paged KV cache
The engine serves from a **paged** KV cache: one array covering the whole pool, carved
into fixed-size pages, with a per-request block table saying which pages hold that
request's tokens. The layout is contract:

- one array of shape `[layers, pages, page_size, 2, kv_heads, head_dim]`, allocated at
  the dtype the run configuration asks for -- not at a dtype you pick;
- **`groups` is a tuple, and for every model in this benchmark it has exactly one
  entry.** That entry is `(num_layers, num_kv_heads, head_dim)` -- the group's first
  field is a SLOT COUNT, and here one slot is one layer, so it equals
  `num_hidden_layers`. Passing one group per layer type-checks, allocates the right
  total memory, and is wrong: it is refused by name at grade time. The plural exists so
  the container can express a family whose layers disagree on `num_kv_heads` or
  `head_dim`, and no model here does.
- the `2` axis is K then V, in that order;
- a token at logical position `pos` in a request whose page list is `page_ids` occupies
  flat slot `page_ids[pos // page_size] * page_size + pos % page_size`, so the block
  table is what makes the layout correct and indexing the cache flat is wrong even when
  the two happen to agree;
- writing into it is functional: the write returns a NEW array and the caller threads it
  onward. Nothing donates or mutates in place.

**Your engine's cache has to be observable from outside.** After a generation run, an
array of exactly the `[layers, pages, page_size, 2, kv_heads, head_dim]` shape above
must be reachable on the `Engine` instance: either directly as an attribute (the
reference calls it `kv_cache`), or one level inside a `list`, `tuple` or `dict`
attribute. A cache buried more than one container deep, held in a closure, or split into
several arrays that are never assembled into one is not reachable, and a correct port
that hides it will still be marked down. Nothing constrains the attribute's NAME.

## API contract

Your repo is imported and driven through the names below, so build to this list. The
module layout is `torch_orig`'s, and the names it does NOT show you take from
`torch_orig` too -- same modules, same classes, same signatures, mirrored exactly as the
Goal says. What this list adds is the part `torch_orig` does not carry faithfully: the
loader's JAX-only checkpoint entry point (it has `load_weights` where you need
`convert_checkpoint`), the cache signatures that moved, the config and result fields the
harness fills by name, and the engine surface it drives. The model internals and the
sampler's transforms are not here on purpose -- their `__call__` matches `torch_orig`'s
`forward` one-to-one, so you take them from there like everything else off this list.

Signatures are shown as the grader CALLS them, with `self` and `cls` omitted. A line
with no parentheses is an attribute read, not a call: `Class.name` is a property or
class attribute, and `Class.NAME: type` must be an ANNOTATED class-level assignment
(`NAME: type = value` in the class body) -- the same name set without an annotation,
assigned on `self` in `__init__`, or inlined at its use reads as absent. Where a name is
load-bearing beyond its signature -- the two `Engine` hooks, the submodule attribute
paths, the caches your engine must leave reachable -- a paragraph elsewhere says so, and
those are the ones to read twice.

```
attention/metadata.py
    AttentionMetadata(input_positions, block_tables, seq_lens, query_start_loc, request_distribution, padded_num_reqs)
    AttentionMetadata.num_seqs
    AttentionMetadata.pages_per_seq
attention/paged.py
    paged_attention(query, kv_cache_layer, attn_metadata, num_kv_heads, sm_scale, sliding_window=None, sinks=None)
cache/block_table.py
    BlockPool(num_pages, page_size)
    BlockPool.num_free
    BlockPool.allocate(num_tokens)
    BlockPool.free(pages)
    BlockPool.slot_mapping(page_ids, start_pos, num_tokens)
    BlockPool.block_table_row(page_ids, pages_per_seq)
cache/kv_cache.py
    make_kv_cache(groups, num_pages, page_size, dtype)
    identity_layer_slot(num_layers)
    identity_layer_writes(num_layers)
    write_kv(cache_layer, key, value, slot_mapping)
config.py
    ModelConfig(hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, vocab_size, max_position_embeddings, rms_norm_eps, rope_theta, rope_scaling, tie_word_embeddings, attention_bias, num_local_experts, num_experts_per_tok, swiglu_limit, sliding_window, layer_types)
    ModelConfig.sliding_window_for_layer(layer_idx)
    ModelConfig.from_hf(model_dir)
    RunConfig(page_size, num_pages, max_num_seqs, pages_per_seq, max_model_len, dtype, seed, num_decode_steps)
    RunConfig.from_yaml(path)
engine.py
    PrefillResult(hidden, logits, slot_mapping, attn_metadata, page_ids, hidden_states)
    DecodeResult(hidden, logits, slot_mapping, attn_metadata)
    GenerateResult(tokens, step_logprobs, logits_per_step)
    Engine(model, run_config)
    Engine.POSITIONS_HOOK
    Engine.INPUTS_HOOK
    Engine.reset()
    Engine.prefill(input_ids, request_id=0, output_hidden_states=False, **model_kwargs)
    Engine.decode_step(request_id, token_id)
    Engine.generate(input_ids, num_steps, params=None, request_id=0, **model_kwargs)
layers/activation.py
    ClampedSwiGLU(limit)
    ClampedSwiGLU.ALPHA: float
    ClampedSwiGLU.__call__(x)
model/loader.py
    DATA_ROOT
    STACKED_PARAMS
    resolve_model_dir(model_name_or_path)
    convert_checkpoint(model_dir, config)
    apply_params(model, params)
    load_tokenizer(model_dir)
modeling.py
    GptOssForCausalLM(config, rngs=None, *, dtype=...)
    GptOssForCausalLM.__call__(input_ids, kv_cache, attn_metadata, slot_mapping=None, output_hidden_states=False)
    GptOssForCausalLM.compute_logits(hidden)
sample/sampler.py
    SamplingParams(temperature, top_k, top_p)
    process_logits(logits, params)
    topn_logprobs(logits, n)
serve.py
    ServeConfig(model_name_or_path, page_size, num_pages, max_num_seqs, pages_per_seq, max_model_len, dtype, seed, num_decode_steps, kv_cache_dtype, prompts, prompt_token_ids, continuations, sampling, do_generate, do_score, output_dir)
    ServeConfig.run_config(*, dtype=None)
```

## Requirements

- **Every directory in your repo must contain Python.** At every depth, a sub-directory
  of `jax_repo/` must hold at least one `.py` file somewhere underneath it -- one with
  none is importable as a namespace package and the structure check refuses it, so put
  prose, shell or other non-Python material in files at the repo root, in a directory
  that also holds Python, or outside `/workspace/jax_repo/` (nothing outside it is
  collected in any case). `__pycache__` is ignored.
- **Convert the checkpoint yourself.** `model/loader.py` exposes `convert_checkpoint`,
  and the engine gets its weights from it. Read the safetensors with
  **`safetensors.flax.load_file`** (or `safetensors.safe_open(..., framework="flax")`):
  `safetensors.torch` and `torch.load` are torch imports even though the file they read
  is a PyTorch checkpoint, and `safetensors.numpy` cannot represent bf16 at all. Follow
  `model.safetensors.index.json` when the checkpoint is sharded.
- **Naming is yours; correspondence is not.** Parameter paths are your own to choose,
  but every leaf must map onto exactly one tensor of the torch checkpoint and carry its
  value -- so keep the module names recognizable (`model.layers.N.self_attn.q_proj`, …);
  the comparison treats `weight`, `kernel`, `embedding` and `scale` as interchangeable
  leaf tokens. Where `torch_orig` FUSES several checkpoint tensors into one parameter
  (the packed QKV and gate/up projections), fuse them in the same order, or the port
  loads clean and computes the wrong thing.
- **Keep `torch_orig`'s submodule ATTRIBUTE names, not just its module and class
  names.** Parts of the grade replay the reference's own activations through one
  submodule at a time, resolving its attribute path against your model object, so the
  path has to exist on your side. On this instance those are
  `model.layers.0.input_layernorm`, `model.layers.0.attn.qkv_proj`,
  `model.layers.0.attn.rotary_emb`, `model.layers.0.attn.o_proj`,
  `model.layers.0.post_attention_layernorm`, `model.layers.0.mlp.router`,
  `model.layers.0.mlp.experts`, `model.norm`, `lm_head`, an all-digit component meaning
  an index. Watch the rotary embedding -- applying RoPE as a free function leaves no
  attribute there, so instantiate `RotaryEmbedding` as a child of your attention module
  and call it, the way `torch_orig` does. This is about the attribute path only; your
  parameter tree is still yours to name.
- **Two hooks live on your `Engine` CLASS, and they are read by name.** Probes that
  drive your MODEL directly rather than through `Engine.prefill` build its
  `input_positions` and extra forward keywords the way your engine does: by reading a
  class attribute off your `Engine` and calling the model method it names.
  `Engine.POSITIONS_HOOK` names the model method that builds `input_positions`;
  `Engine.INPUTS_HOOK` names the one that turns raw forward keywords into ones a traced
  forward can carry. Both are resolved only this way -- an engine that reaches the
  method inline instead gets the DEFAULTS: positions `0, 1, 2, ...` ascending, and
  keywords passed through untouched. Those defaults are correct for a text-only family,
  so you may define neither; define them, and the methods behind them, when they are not
  -- a `(t, h, w)` position triple, or a keyword a vision tower must consume on the HOST
  because its geometry is Python control flow that cannot run under `jax.jit`. A missing
  hook is not diagnosed as missing -- it silently takes the default.
- **Synthesize what the checkpoint does not carry, IN `convert_checkpoint`.** When the
  config ties the output head, the checkpoint holds one tensor but torch's
  `state_dict()` holds two, and the comparison is against the state dict -- so
  `convert_checkpoint` must RETURN both `lm_head` and the embedding with the same
  values; tying them inside the model leaves the tree one entry short. The rule is
  `state_dict()`, not everything the file omits: `layers/rotary.py` registers the rotary
  tables with `persistent=False`, so they are absent from `state_dict()` and must NOT
  appear in what `convert_checkpoint` returns. Build them inside your model, in float32
  -- rounding them to the compute dtype is a silent precision loss.
- **The paged cache and the block table are the point of the task.** Honour the layout
  in the instance contract above, allocate pages through the pool rather than assuming a
  request owns a contiguous range, and gather through the block table on every read.
  Attention must read only the pages the block table lists and only up to the sequence
  length -- a query must never see a slot belonging to a later position or to another
  request.
- **Static shapes and one compilation per shape.** The graded loop calls prefill once
  and then decodes step by step; each of those must reuse a single compiled program
  rather than retracing per call, so pad to fixed buckets and keep every
  shape-determining quantity out of the traced arguments. The forward must trace under
  `jax.jit` with no host callbacks, and no Python control flow may branch on a traced
  value.
- **Do not hard-code a compute dtype.** It comes from the run configuration, and the
  model has to compute correctly in whatever dtype it is handed -- including a
  full-precision run, which is graded far more tightly than the serving one.
- **Keep the sampler's transforms exact.** Temperature, top-k and top-p are
  deterministic functions of the logits and are compared as such; the greedy choice and
  the top-N log-probabilities likewise. No RNG draw is compared across frameworks, so a
  differently-seeded sampler is fine -- a differently-SHAPED transform is not.
- **Everything crossing your API boundary is a `jax.Array` or a plain Python value** --
  never a `torch.Tensor`, and never a numpy array standing in for a device array.
- **Load the tokenizer with HuggingFace `tokenizers`.** It is the torch-free tokenizer
  library the offline grading container ships -- `transformers` is absent for the same
  reason `torch` is, and a `pip install` of anything else fails at grade time. Read the
  checkpoint's own `tokenizer.json` from the directory named above with
  `tokenizers.Tokenizer.from_file`, not by repo id.
- **Ship a working `serve.py`.** It is the one entry point that composes the whole repo
  from a config dict: resolve the model directory, load the tokenizer, build the model
  at the config's dtype, convert and apply the checkpoint, drive the engine, decode to
  text, and write `serve_results.json`. Two modes off config flags -- `do_generate`
  (free-running greedy decode) and `do_score` (teacher-forced scoring of a supplied
  continuation). Under `do_score`, the log-probability for continuation token `t` is
  read from the logits at position `prompt_len + t - 1`, the row that predicts it.
  `kv_cache_dtype` is separate from the compute dtype and defaults to it; honour it when
  the config names one. The graded checks drive `serve.py` with `prompt_token_ids`, so
  the text-prompt path is a convenience; `do_score`'s per-token log-probs and the
  decoded `text` are graded exactly.

## How you are graded

`jax_repo/` is graded in an offline, torch-blocked container against reference artifacts
frozen from a run of `torch_orig` on a fixed serving scenario. Grading is a **continuous
fidelity score**, not pass/fail: the closer your port comes to reproducing the reference,
the higher it scores (even a faithful fp32 port lands just short of the top -- bit-for-bit
is not expected). Structure (`spec`) is the gate -- it multiplies the whole, so any spec
failure scales your score down; `numeric` and `behavioral` then score how closely your
port matches, at equal weight. Three dimensions:

- **spec** -- independence and structure: no torch anywhere in the tree, the module and
  symbol layout above, a cache and block table that match the contract, a forward that
  traces to real XLA ops and compiles once per shape (not once per call), a parameter set
  that corresponds to the checkpoint's and carries its values, and the serving-config
  defaults your `config.py` resolves under no overrides.
- **numeric** -- how closely your forward matches the reference: the prefill logits, the
  decode-step logits, the hidden states, the cache contents after a write, the isolated
  attention operation, individual submodules taken one at a time, and the sampler
  transforms -- graded at two precisions, the serving dtype named above and a
  full-precision run held to a far tighter band. On top of this single frozen scenario, a
  set of generalisation batteries each varies ONE axis off it and compares the result
  head-on to the reference: the prompt length and the physical page assignment, plus --
  where the mode has them -- the prefill/decode composition of a mixed batch and the split
  point of a chunked prefill. A port that reproduces only the one frozen scenario but does
  not generalise fails the axis it breaks, and a broken axis counts against you out of
  proportion to its share of the checks -- you cannot average it away behind the many
  passing single-shot parity checks.
- **behavioral** -- the engine run for real: a free-running generation reproduces the
  reference's tokens and their likelihoods, decode agrees with prefill on the same prefix,
  held-out prompts match, and the invariants a paged server must hold all hold -- the same
  logical sequence gives the same logits under a different physical page assignment,
  reusing a freed page does not resurrect its old contents, repeated runs are
  deterministic, sequences do not leak into one another, and -- where the mode runs them
  -- batching and chunking a request do not change its result.
