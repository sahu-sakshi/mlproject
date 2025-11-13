# XLM-RoBERTa Fake News Detection Model

This folder contains the complete implementation for the XLM-RoBERTa-based fake news detection model with domain generalization techniques.

## 📁 Folder Structure

```
xlm_roberta_fake_news/
├── run_training.py          # Main training script (self-contained, recommended)
├── main.py                  # Alternative training script (uses data_loader.py)
├── train_model.py           # Model classes and training logic
├── data_loader.py           # Dataset loading utilities
├── example_train.sh         # Example training commands
├── TRAINING_OVERVIEW.md     # Detailed documentation
├── README.md                # This file
└── output/                  # Training outputs and saved models
    ├── best_model/          # Saved model weights
    ├── training_history.json
    └── test_results.json
```

## 🚀 Quick Start

### Training the Model

From this folder:
```bash
python run_training.py
```

Or from project root:
```bash
cd models/xlm_roberta_fake_news
python run_training.py
```

### With Custom Options
```bash
python run_training.py \
  --model_name xlm-roberta-base \
  --batch_size 16 \
  --num_epochs 3 \
  --use_lora \
  --use_rdrop \
  --use_groupdro \
  --output_dir output
```

## 📊 Model Details

### Base Model
- **XLM-RoBERTa-base** (default) or BERT-multilingual-cased
- Multilingual transformer (100+ languages)
- 125M parameters

### Features
- ✅ LoRA fine-tuning (parameter-efficient)
- ✅ R-Drop regularization
- ✅ Mixup augmentation
- ✅ Domain generalization (GroupDRO, IRM, DANN)
- ✅ Multilingual support (English + Hindi)

### Datasets Used
- `english_fake_news_2212.csv`
- `fake_news_dataset.csv`
- `dataset-merged.csv` (Hindi + English)

**Note**: Datasets are located in `../../dataset/` (project root)

## 📈 Training Configuration

### Default Hyperparameters
- Learning Rate (PLM): 2e-5
- Learning Rate (Head): 1e-4
- Batch Size: 16
- Epochs: 3
- Max Length: 512 tokens
- Gradient Accumulation: 2 steps

### Data Splits
- Train: 70%
- Validation: 10%
- Test: 20%
- Stratified by label and domain

## 📤 Output Files

After training, check the `output/` folder:
- `best_model/` - Saved model weights (HuggingFace format)
- `training_history.json` - Training curves and metrics
- `test_results.json` - Final test set evaluation

## 🔧 Dependencies

See `../../requirements.txt` for all dependencies.

Main packages:
- torch >= 2.0.0
- transformers >= 4.35.0
- peft >= 0.6.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0

## 📝 Notes

- Dataset path is relative: `../../dataset/` (from model folder)
- Output is saved in `output/` (relative to model folder)
- Model automatically uses CUDA if available
- All random seeds are fixed for reproducibility

## 🔍 For More Details

See `TRAINING_OVERVIEW.md` for comprehensive documentation.

