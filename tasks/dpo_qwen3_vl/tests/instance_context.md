## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `Qwen/Qwen3-VL-4B-Instruct`
- `template`: `qwen3_vl_nothink`
- training method: `dpo`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text + vision

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/Qwen/Qwen3-VL-4B-Instruct` — `config.json`, the tokenizer files, and the weights.
Nothing is downloaded at train time or at grade time, so this directory is the only
copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: 2 shards listed in `model.safetensors.index.json` (713 tensors, 8.3 GiB), all BF16
- no `lm_head` tensor in the file: the output head is tied to the embedding table

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[2, 88] dtype=int64
  image_grid_thw: shape=[1, 3] dtype=int64
  input_ids: shape=[2, 88] dtype=int64
  labels: shape=[2, 88] dtype=int64
  pixel_values: shape=[256, 1536] dtype=float32
  position_ids: shape=[3, 2, 88] dtype=int64
  rope_deltas: shape=[2, 1] dtype=int64
```
