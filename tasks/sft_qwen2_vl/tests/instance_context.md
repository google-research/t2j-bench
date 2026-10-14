## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `Qwen/Qwen2-VL-2B-Instruct`
- `template`: `qwen2_vl`
- training method: `sft`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text + vision

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/Qwen/Qwen2-VL-2B-Instruct` — `config.json`, the tokenizer files, and the weights.
Nothing is downloaded at train time or at grade time, so this directory is the only
copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: 2 shards listed in `model.safetensors.index.json` (729 tensors, 4.1 GiB), all BF16
- no `lm_head` tensor in the file: the output head is tied to the embedding table

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[1, 20] dtype=int64
  image_grid_thw: shape=[1, 3] dtype=int64
  input_ids: shape=[1, 20] dtype=int64
  labels: shape=[1, 20] dtype=int64
  pixel_values: shape=[16, 1176] dtype=float32
  position_ids: shape=[3, 1, 20] dtype=int64
  rope_deltas: shape=[1, 1] dtype=int64
```
