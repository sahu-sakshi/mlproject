# Optimized mDeBERTa-v3-base Fake News Detection

This folder contains an optimized mDeBERTa-v3-base model for fake news detection targeting **93%+ accuracy**.

## Features

- **Base Model**: microsoft/mdeberta-v3-base
- **LoRA Fine-tuning**: Parameter-efficient training with rank=16
- **Regularization**: R-Drop and Mixup augmentation
- **Domain Generalization**: GroupDRO and DANN
- **Mixed Precision**: FP16 training for efficiency
- **Multi-dataset**: Supports English, Kannada, Tamil, and Malayalam datasets

## Quick Start

### Basic Training (with all optimizations)

```bash
cd models/deberta_optimised
python run.py \
  --model_name microsoft/mdeberta-v3-base \
  --batch_size 16 \
  --num_epochs 4 \
  --max_length 512 \
  --use_lora \
  --lora_rank 16 \
  --use_rdrop \
  --use_mixup \
  --use_groupdro \
  --use_dann \
  --lr_plm 1e-5 \
  --lr_head 1e-4 \
  --gradient_accumulation_steps 2 \
  --use_fp16 \
  --output_dir output/93_percent_run
```

### Default Training (recommended)

The script uses optimized defaults from requirements.md:

```bash
python run.py
```

This will use:
- Model: mdeberta-v3-base
- Epochs: 4
- Max length: 512
- LoRA rank: 16
- Learning rate (PLM): 1e-5
- FP16: Enabled
- All regularization and domain generalization techniques

## Datasets Used

The script automatically loads:
- `english_fake_news_2212.csv`
- `train (2).csv` and `test (1).csv`
- `ka_true.csv` and `ka_fake.csv` (Kannada)
- `ta_true.csv` and `ta_fake.csv` (Tamil)
- `ma_true.csv` and `ma_fake.csv` (Malayalam)
- `LabeledAuthentic-7K.csv` and `LabeledFake-1K.csv`
- `Authentic-48K.csv` (if available)

All datasets are located in `../../dataset/` (relative to this folder).

## Output

After training, check:
- `output/93_percent_run/best_model/` - Saved model weights
- `output/93_percent_run/training_history.json` - Training metrics
- `output/93_percent_run/test_results.json` - Final test metrics

The script will verify if 93%+ accuracy is achieved and display a success message.

## Model Architecture

- **Base**: mDeBERTa-v3-base (multilingual transformer)
- **LoRA Target Modules**: `["query_proj", "key_proj", "value_proj", "dense"]`
- **Classification Head**: Binary classification (Fake=1, Real=0)

## Training Configuration

### Default Hyperparameters
- Learning Rate (PLM): 1e-5
- Learning Rate (Head): 1e-4
- Batch Size: 16
- Epochs: 4
- Max Length: 512
- LoRA Rank: 16
- Gradient Accumulation: 2
- Warmup Steps: 500
- Weight Decay: 0.01

### Regularization
- **R-Drop**: α=2.0 (KL divergence consistency)
- **Mixup**: α=0.2 (data augmentation)

### Domain Generalization
- **GroupDRO**: η=0.01 (group distributionally robust optimization)
- **DANN**: λ=1.0 (domain adversarial neural network)

## Requirements

See `../../requirements.txt` for dependencies. Key packages:
- torch
- transformers
- peft (for LoRA)
- scikit-learn
- pandas
- numpy

## Notes

- The model uses FP16 mixed precision by default for faster training
- LoRA significantly reduces memory usage while maintaining performance
- All datasets are automatically preprocessed and combined
- Stratified splitting ensures balanced train/val/test sets

