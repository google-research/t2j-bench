## Instance context

This task targets one specific model, checkpoint and serving mode. The sections below
give the model identity, the checkpoint you must convert, and the shape of the paged KV
cache your engine has to maintain; everything else about the repo layout mirrors
`torch_orig`.

### Model
- `model_name_or_path`: `google/gemma-4-E2B-it`
- serving mode: `ragged` -- a persistent batch of concurrent requests, addressed by `query_start_loc` rather than by a batch dimension.
- graded precision profile: `bf16_native` (dtype `bfloat16`)

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/google/gemma-4-E2B-it` -- `config.json`, the tokenizer files, and the weights.
Nothing is downloaded at any point: the container has no network at grade time, this
directory is the only copy, and your `model/loader.py` reads it straight from there. The
path is a hard contract, not a default -- `resolve_model_dir` resolves
`/data/{model_name_or_path}` and the grading image bakes the same bytes the agent image
does.

**This checkpoint is tri-modal and you port the TEXT DECODER ONLY.**
`google/gemma-4-E2B-it` declares `architectures: ["Gemma4ForConditionalGeneration"]` and
its `config.json` carries a `vision_config` and an `audio_config` beside the
`text_config`. `torch_orig/` implements the text decoder and nothing else, and so must
your port: there is no vision tower, no audio tower and no modality projection in the
reference, and none is graded.

**The decoder's own shape lives under `config.json`'s `text_config`, not at its top
level.** Read it from there. The top level describes the conditional-generation wrapper
-- the towers, the modality token ids, `tie_word_embeddings` -- and reading
`num_hidden_layers` or `hidden_size` from it will simply not find them.

**The text tensors are stored under `model.language_model.*`**, beside
`model.vision_tower.*` and `model.audio_tower.*`, because the whole bundle shares one
file. Your converter names the tensors it wants, so the tower tensors are never read;
that is not an oversight and there is no coverage rule that requires reading them. Of
the 2011 tensors in `model.safetensors`, 600 are the text decoder's.

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
The engine serves from a **paged** KV cache: a pool carved into fixed-size pages, with a
per-request block table saying which pages hold that request's tokens. The layout is
contract:

- **this family's cache is MORE THAN ONE ARRAY**, one per distinct KV geometry, each of
  shape `[n_slots, pages, page_size, 2, kv_heads, head_dim]` and each allocated at the
  dtype the run configuration asks for -- not at a dtype you pick. `head_dim` differs
  between this model's two attention types, so no single rectangular array can hold it;
- **`groups` is a tuple and for this model it has MORE THAN ONE entry, and the slot
  counts do NOT sum to `num_hidden_layers`.** Each entry is `(n_slots, num_kv_heads,
  head_dim)`. Two things follow that are true here and false for every other model in
  this benchmark. First, layers of DIFFERENT attention types have different `head_dim`,
  so they land in different groups -- read both values off `config.json`'s
  `text_config`, which declares them separately, and confirm them against the
  checkpoint's own projection shapes. Second, `num_kv_shared_layers` of the LAST layers
  do not allocate a slot at all: each attends against the cache of the last preceding
  layer of the SAME attention type, and must not write to it. So a group's `n_slots` is
  a count of ALLOCATING layers, and the layer index is not the slot index. Allocating
  one slot per layer type-checks, allocates more memory than the model needs, and is
  wrong.
- the `2` axis is K then V, in that order;
- a token at logical position `pos` in a request whose page list is `page_ids` occupies
  flat slot `page_ids[pos // page_size] * page_size + pos % page_size`, so the block
  table is what makes the layout correct and indexing the cache flat is wrong even when
  the two happen to agree;
- writing into it is functional: the write returns a NEW array and the caller threads it
  onward. Nothing donates or mutates in place.

**Your engine's cache has to be observable from outside.** After a generation run, the
arrays described above must be reachable on the `Engine` instance: either directly as an
attribute holding the tuple (the reference calls it `kv_cache`), or one level inside a
`list`, `tuple` or `dict` attribute. A cache buried more than one container deep or held
in a closure is not reachable, and a correct port that hides it will still be marked
down. Nothing constrains the attribute's NAME. Note that this family is the exception to
the usual rule that the cache is one array: here several arrays IS the contract, and
they must NOT be padded to a common `head_dim` and concatenated.
