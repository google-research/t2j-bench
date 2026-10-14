## Instance context

This task targets one specific model, checkpoint and serving mode. The sections below
give the model identity, the checkpoint you must convert, and the shape of the paged KV
cache your engine has to maintain; everything else about the repo layout mirrors
`torch_orig`.

### Model
- `model_name_or_path`: `openai/gpt-oss-20b`
- serving mode: `chunked` -- a persistent batch of concurrent requests, exactly as `ragged` is, plus two things `ragged` does not have.
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

**And in this mode the PREFIX cache has to be observable too -- under that exact name.**
The object your engine shares pages through is read from outside as
`Engine.prefix_cache`: a plain public attribute on the `Engine` instance, spelled
exactly that way, with no leading underscore. Unlike the KV cache one paragraph up, this
one is NOT found by shape and no other spelling is tried. Whatever it holds must answer
`.stats()` with a mapping carrying a `"hits"` key whose value is an integer count of the
prefix-cache hits served so far, and that count must actually MOVE when a prefill is
served from a shared prefix and stay put when it is not -- it is read before and after a
prefill and the difference is what says a hit happened. A port that keeps the cache
private, or reports a count that never changes, is marked down even though its logits
are right.
