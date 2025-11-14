# mDeBERTa-v3 Fake News Detection - Training Results

**Training Date:** Training completed

## Model Information

- **Base Model:** microsoft/mdeberta-v3-base
- **Task:** Binary Classification (Fake/True News)
- **Languages:** Kannada, English

## Dataset Information

- **Total Samples:** 8,490
- **Kannada Samples:** 6,278 (Fake: 3,220, True: 3,058)
- **English Samples:** 2,212 (Fake: 1,110, True: 1,102)
- **Fake News Samples:** 4,330
- **True News Samples:** 4,160
- **Train Set:** 6,792 (80%)
- **Validation Set:** 1,698 (20%)

### Dataset Files Used
- `ka_fake.csv` - 3,220 samples
- `ka_true.csv` - 3,058 samples
- `english_fake_news_2212.csv` - 2,212 samples

## Training Configuration

- **Epochs:** 3
- **Batch Size:** 8
- **Learning Rate:** 2e-5
- **Weight Decay:** 0.01
- **Warmup Steps:** 500
- **Max Length:** 256
- **Mixed Precision (FP16):** True

## Evaluation Results

- **Accuracy:** 0.8233 (82.33%)
- **F1 Score:** 0.8233 (82.33%)
- **Validation Loss:** 0.4019

## Test Inference Examples

*Results will appear here after training completes*

## Model Output

- **Model Saved To:** mdeberta_fakenews_model/

---

*Training completed successfully. Model achieved 82.33% accuracy and F1 score on validation set.*

