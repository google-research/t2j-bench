## Instance context

This task targets one specific model and checkpoint. The sections below give the model
identity, the checkpoint you must convert, and the exact batch schema your data pipeline
must reproduce; everything else about the repo layout mirrors `torch_orig`.

### Model
- `model_name_or_path`: `meta-llama/Llama-3.2-1B-Instruct`
- `template`: `llama3`
- training method: `dpo`
- finetuning: `lora` — rank 8, alpha 16, all linear projections (base frozen; only adapters train)
- graded precision profile: `fp32_native` (dtype `float32`)
- modalities: text

### The checkpoint you convert
The HF checkpoint for the `model_name_or_path` above is baked into the image at
`/data/meta-llama/Llama-3.2-1B-Instruct` — `config.json`, the tokenizer files, and the
weights. Nothing is downloaded at train time or at grade time, so this directory is the
only copy, and your `model.loader.convert_checkpoint` reads it straight from there.
- weights: a single `model.safetensors` (146 tensors, 2.3 GiB), all BF16
- no `lm_head` tensor in the file: the output head is tied to the embedding table

### Batch schema (your data pipeline must reproduce these keys/shapes)
```
  attention_mask: shape=[2, 16] dtype=int64
  input_ids: shape=[2, 16] dtype=int64
  labels: shape=[2, 16] dtype=int64
```
