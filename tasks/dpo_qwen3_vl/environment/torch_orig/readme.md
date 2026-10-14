# Qwen3-VL SFT Framework
Extracted from LLamaFactory.

### Usage
```bash
cd high_level/sft/pytorch
python train.py config.yaml
# With overrides:
python train.py config.yaml learning_rate=5e-6 num_train_epochs=1
```