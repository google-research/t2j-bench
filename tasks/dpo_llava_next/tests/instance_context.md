## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `llava-hf/llava-v1.6-vicuna-7b-hf`
- `template`: `llava_next`
- training method: `dpo`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text + vision

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/llava-hf/llava-v1.6-vicuna-7b-hf` — `config.json`, the tokenizer files, and the
weights. Nothing is downloaded at train time or at grade time, so this directory is the
only copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: 3 shards listed in `model.safetensors.index.json` (687 tensors, 13.2 GiB), all FP16
- the file carries its own output head, `language_model.lm_head.weight`

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[2, 1232] dtype=int64
  image_sizes: shape=[1, 2] dtype=int64
  input_ids: shape=[2, 1232] dtype=int64
  labels: shape=[2, 1232] dtype=int64
  pixel_values: shape=[1, 3, 3, 336, 336] dtype=float32
```
