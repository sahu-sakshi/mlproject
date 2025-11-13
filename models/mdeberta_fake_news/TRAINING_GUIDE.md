# mDeBERTa-v3 Training Guide

## Overview

This script fine-tunes `microsoft/mdeberta-v3-base` for multilingual fake news detection using:
- **Kannada datasets**: `ka_fake.csv` and `ka_true.csv`
- **English dataset**: `english_fake_news_2212.csv`

## Dataset Structure

### Kannada Datasets
- **Columns**: `text`, `label`
- **Labels**: Already encoded (0=fake, 1=true)

### English Dataset
- **Columns**: `news_id`, `headline`, `body_text`, `source`, `label`
- **Labels**: 'Real' or 'Fake' (mapped to 1 and 0 respectively)
- **Text**: Combined `headline + body_text`

## Training Process

1. **Data Loading**: Reads CSV files from `../../dataset/`
2. **Preprocessing**: 
   - Combines all datasets
   - Removes empty texts
   - Shuffles data
3. **Splitting**: 80% training, 20% validation
4. **Tokenization**: Max length 256, truncation and padding
5. **Training**: 3 epochs with fp16 mixed precision
6. **Evaluation**: Accuracy and F1 score after each epoch
7. **Saving**: Best model saved to `mdeberta_fakenews_model/`
8. **Inference**: Test examples at the end

## Usage

```bash
cd models/mdeberta_fake_news
python run_training_kan_eng.py
```

## Output Files

- `mdeberta_fakenews_model/` - Saved model and tokenizer
- `training_output/` - Training checkpoints
- `logs/` - Training logs

## Model Details

- **Base Model**: microsoft/mdeberta-v3-base
- **Task**: Binary classification (Fake=0, True=1)
- **Languages**: Kannada and English
- **Max Sequence Length**: 256 tokens

## Training Hyperparameters

- Learning Rate: 2e-5
- Batch Size: 8
- Epochs: 3
- Optimizer: AdamW (default)
- Mixed Precision: fp16
- Best Model Selection: Based on F1 score

