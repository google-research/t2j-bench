# Gemma3 SFT Pytest Suite

Pytest suite that verifies the JAX repo (`jax_repo/`) produces identical results to the reference PyTorch repo (`torch_orig/`) across all modules.

## Test Structure

| File | Module Under Test | What It Verifies |
|------|-------------------|------------------|
| `test_configuration.py` | `transformers/configuration.py` | PreTrainedConfig, SiglipVisionConfig, Gemma3TextConfig, Gemma3Config, config loading |
| `test_modeling.py` | `transformers/modeling.py` | RMSNorm, embeddings, RoPE, attention, SigLip vision encoder, multimodal projector, full model forward & loss |
| `test_utils.py` | `transformers/utils.py`, `data/constants.py`, `data/tool_utils.py`, `data/mm_plugin.py` | ACT2FN activations, ModelOutput classes, BatchFeature, constants, tool utils, MM plugin |
| `test_formatter.py` | `data/formatter.py` | EmptyFormatter, StringFormatter, FunctionFormatter, ToolFormatter |
| `test_template.py` | `data/template.py` | Template registry, chat encoding (single/multi-turn), system message fusion, EOS replacement |
| `test_data_processor.py` | `data/processor.py` | `infer_seqlen`, SupervisedDatasetProcessor (label masking, train_on_prompt, mask_history, cutoff truncation) |
| `test_collator.py` | `data/collator.py` | `prepare_4d_attention_mask`, MultiModalDataCollatorForSeq2Seq, SFTDataCollatorWith4DAttentionMask |
| `test_model_loader.py` | `model/loader.py` | `_cross_entropy_loss`, `_to_torch_tensor`, `_as_param_spec`, `_state_dict_to_param_specs`, TokenizerModule |

Shared fixtures and dynamic module loading are defined in `conftest.py`.

## Prerequisites

- Conda environment `lf` with `pytest`, `torch`, `jax`, and `numpy` installed.

## How to Run

Run all tests:

```bash
conda run -n lf python -m pytest dataset/sft/gemma3/pytest/ -v
```

Run a single test file:

```bash
conda run -n lf python -m pytest dataset/sft/gemma3/pytest/test_collator.py -v
```

Run a specific test class or method:

```bash
# Run all tests in a class
conda run -n lf python -m pytest dataset/sft/gemma3/pytest/test_utils.py::TestActivationFunctions -v

# Run a single test
conda run -n lf python -m pytest dataset/sft/gemma3/pytest/test_utils.py::TestActivationFunctions::test_act2fn_keys_match -v
```

Run with keyword filtering:

```bash
conda run -n lf python -m pytest dataset/sft/gemma3/pytest/ -k "collator" -v
```
