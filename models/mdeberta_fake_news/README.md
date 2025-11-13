# mDeBERTa-v3 Multilingual Fake News Detection

Fine-tuned `microsoft/mdeberta-v3-base` for multilingual fake news detection using Kannada and English datasets.

## 📊 Datasets

- **Kannada**: `ka_fake.csv` and `ka_true.csv`
- **English**: `english_fake_news_2212.csv`

## 🚀 Quick Start

### Training

```bash
cd models/mdeberta_fake_news
python run_training_kan_eng.py
```

The script will:
1. Load and combine Kannada and English datasets
2. Preprocess and split (80% train, 20% validation)
3. Fine-tune mDeBERTa-v3-base
4. Save model to `mdeberta_fakenews_model/`
5. Run test inference on example headlines

## ⚙️ Training Configuration

- **Model**: microsoft/mdeberta-v3-base
- **Learning Rate**: 2e-5
- **Batch Size**: 8
- **Epochs**: 3
- **Max Length**: 256 tokens
- **Mixed Precision**: fp16 enabled
- **Best Model**: Loaded based on F1 score

## 📈 Evaluation Metrics

- Accuracy
- F1 Score (weighted)

## 📁 Output

- **Model**: `mdeberta_fakenews_model/` - Saved model and tokenizer
- **Training Logs**: `training_output/` - Checkpoints and logs
- **Logs**: `logs/` - Training logs

## 🔍 Test Inference

The script includes test inference with example headlines at the end to verify the model works correctly.

## 📝 Notes

- Dataset path: `../../dataset/` (relative to model folder)
- Labels: Fake=0, True=1
- Model supports both Kannada and English text

