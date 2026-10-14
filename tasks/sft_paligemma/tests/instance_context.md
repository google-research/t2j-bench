## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `google/paligemma-3b-pt-224`
- `template`: `paligemma`
- training method: `sft`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text + vision

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/google/paligemma-3b-pt-224` — `config.json`, the tokenizer files, and the
weights. Nothing is downloaded at train time or at grade time, so this directory is the
only copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: 3 shards listed in `model.safetensors.index.json` (603 tensors, 10.9 GiB), all FP32
- no `lm_head` tensor in the file: the output head is tied to the embedding table

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[1, 272] dtype=int64
  input_ids: shape=[1, 272] dtype=int64
  labels: shape=[1, 272] dtype=int64
  pixel_values: shape=[1, 3, 224, 224] dtype=float32
  token_type_ids: shape=[1, 272] dtype=int64
```
