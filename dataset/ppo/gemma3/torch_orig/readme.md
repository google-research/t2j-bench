# Gemma3 PPO Training Pipeline

Standalone PPO training pipeline for Gemma3 multimodal models, decoupled from HuggingFace transformers and TRL.

## Usage

```bash
python train.py examples/gemma3_ppo.yaml
```

## Structure

```
├── train.py                 # PPO entry point
├── config.py                # Configuration system (YAML/CLI)
├── model/
│   └── loader.py            # Model + value head loading
├── data/
│   ├── loader.py            # Dataset loading (prompt-only for PPO)
│   ├── processor.py         # UnsupervisedDatasetProcessor
│   ├── collator.py          # Multimodal data collator
│   ├── template.py          # Gemma3 chat templates
│   ├── mm_plugin.py         # Gemma3 image processing
│   ├── formatter.py         # Template formatters
│   ├── tool_utils.py        # Tool calling utilities
│   └── constants.py         # Constants
├── ppo/
│   ├── trainer.py           # Standalone PPO trainer
│   └── ppo_utils.py         # PPO algorithm (GAE, losses)
├── datasets/
│   └── dataset.py           # Minimal dataset implementation
├── transformers/            # Standalone Gemma3 model (PyTorch only)
│   ├── modeling.py          # Full Gemma3 architecture
│   ├── configuration.py     # Model configurations
│   ├── processing.py        # Gemma3Processor
│   ├── image_processing.py  # Image preprocessing
│   ├── tokenization.py      # Tokenizer
│   ├── ppo_compat.py        # GenerationConfig, schedulers
│   ├── collator.py          # Base data collator
│   ├── trainer.py           # Base trainer
│   └── utils.py             # Activation functions, utilities
└── examples/
    └── gemma3_ppo.yaml      # Example config
```
