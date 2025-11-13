# Models Directory

This directory contains separate folders for each trained model. Each model folder is self-contained with its own training scripts, model definitions, and results.

## 📁 Current Models

### `xlm_roberta_fake_news/`
XLM-RoBERTa-based fake news detection model with domain generalization techniques.

**Features:**
- Multilingual support (English + Hindi)
- Domain generalization (GroupDRO, IRM, DANN)
- LoRA fine-tuning
- R-Drop and Mixup regularization

See `xlm_roberta_fake_news/README.md` for details.

## 🗂️ Folder Structure

```
models/
├── README.md                    # This file
├── xlm_roberta_fake_news/       # Model 1: XLM-RoBERTa
│   ├── run_training.py
│   ├── train_model.py
│   ├── README.md
│   ├── TRAINING_OVERVIEW.md
│   └── output/
│       ├── best_model/
│       ├── training_history.json
│       └── test_results.json
└── [future_model_2]/            # Model 2: (to be added)
    └── ...
```

## 📊 Shared Resources

- **Datasets**: Located at `../dataset/` (project root)
- **Requirements**: Located at `../requirements.txt` (project root)

Each model folder references these shared resources using relative paths.

## ➕ Adding a New Model

1. Create a new folder: `models/your_model_name/`
2. Copy or create your training scripts
3. Update dataset path to `../../dataset/`
4. Create a `README.md` in your model folder
5. Train and save outputs to `output/` within your model folder

## 🎯 Benefits of This Structure

- ✅ **Isolation**: Each model is self-contained
- ✅ **Organization**: Easy to find and compare models
- ✅ **Reusability**: Datasets shared, no duplication
- ✅ **Clarity**: Each model has its own documentation
- ✅ **Scalability**: Easy to add more models

