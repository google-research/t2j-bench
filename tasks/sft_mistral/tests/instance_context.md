## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `mistralai/Mistral-7B-Instruct-v0.3`
- `template`: `mistral`
- training method: `sft`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/mistralai/Mistral-7B-Instruct-v0.3` — `config.json`, the tokenizer files, and the
weights. Nothing is downloaded at train time or at grade time, so this directory is the
only copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: 3 shards listed in `model.safetensors.index.json` (291 tensors, 13.5 GiB), all BF16
- the file carries its own output head, `lm_head.weight`

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[1, 17] dtype=int64
  input_ids: shape=[1, 17] dtype=int64
  labels: shape=[1, 17] dtype=int64
```
