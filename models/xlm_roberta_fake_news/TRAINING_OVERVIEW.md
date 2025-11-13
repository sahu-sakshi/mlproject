# Fake News Detection Training Pipeline - Complete Overview

## 📊 Datasets Used

The pipeline combines **3 datasets** for multilingual fake news detection:

### 1. **english_fake_news_2212.csv**
- **Format**: Headline + Body Text
- **Labels**: 'Real' → 0, 'Fake' → 1
- **Language**: English
- **Domain**: Extracted from 'source' column
- **Text**: `headline + " " + body_text`

### 2. **fake_news_dataset.csv**
- **Format**: Title + Text
- **Labels**: 'real' → 0, 'fake' → 1
- **Language**: English
- **Domain**: Extracted from 'source' column
- **Text**: `title + " " + text`

### 3. **dataset-merged.csv**
- **Format**: Direct text column
- **Labels**: Integer (0 = Real, 1 = Fake)
- **Language**: Hindi + English (auto-detected via Devanagari script)
- **Domain**: 'merged'
- **Language Detection**: Uses regex `[\u0900-\u097F]` to detect Hindi

### Data Preprocessing
- Lowercasing all text
- Removing URLs (http/https/www)
- Normalizing whitespace
- Removing duplicates based on text content
- Combining all datasets into single DataFrame

### Data Splits
- **Train**: 70% (default)
- **Validation**: 10% (default)
- **Test**: 20% (default)
- **Stratification**: By label AND domain (ensures balanced distribution)

---

## 🤖 Model Architecture

### Base Models Supported
1. **XLM-RoBERTa-base** (default)
   - Multilingual transformer (100+ languages)
   - 125M parameters
   - Best for multilingual tasks

2. **BERT-base-multilingual-cased**
   - Alternative multilingual option
   - 110M parameters

### Model Components

#### **FakeNewsModel** (`train_model.py`)
```python
- Backbone: XLMRobertaForSequenceClassification or BertForSequenceClassification
- Classification Head: 2 classes (Real=0, Fake=1)
- Optional LoRA: Parameter-efficient fine-tuning
```

#### **LoRA (Low-Rank Adaptation)**
- **Rank**: 8 (default)
- **Alpha**: 16
- **Dropout**: 0.1
- **Target Modules**: 
  - RoBERTa: ["query", "value", "key_proj", "dense"]
  - BERT: ["query", "value", "key", "dense"]
- **Benefit**: Trains only ~1% of parameters, faster training

---

## 🎯 Training Techniques

### 1. **Regularization Methods**

#### **R-Drop** (Regularized Dropout)
- **Purpose**: Consistency regularization
- **How**: Two forward passes with same input, KL divergence penalty
- **Alpha**: 2.0 (default)
- **Loss**: `CE_loss + alpha * KL_divergence`

#### **Mixup**
- **Purpose**: Data augmentation via interpolation
- **How**: Mixes two samples: `λ * sample_a + (1-λ) * sample_b`
- **Alpha**: 0.2 (default, Beta distribution parameter)
- **Benefit**: Improves generalization

### 2. **Domain Generalization Techniques**

#### **GroupDRO** (Group Distributionally Robust Optimization)
- **Purpose**: Handle domain shift by upweighting worst-performing groups
- **How**: 
  - Tracks loss per domain group
  - Computes exponential weights: `exp(η * group_loss)`
  - Normalizes and applies to per-sample losses
- **Eta**: 0.01 (learning rate for group weights)

#### **IRM** (Invariant Risk Minimization)
- **Purpose**: Learn domain-invariant features
- **How**: Penalizes gradient of loss w.r.t. dummy classifier scale
- **Penalty Weight**: 1e3 (default)
- **Benefit**: Forces model to learn features that work across domains

#### **DANN** (Domain Adversarial Neural Network)
- **Purpose**: Adversarial domain adaptation
- **How**: 
  - Domain classifier tries to predict source domain
  - Feature extractor tries to fool domain classifier
  - Adversarial loss: `λ * domain_classification_loss`
- **Lambda**: 1.0 (default)
- **Architecture**: 2-layer MLP (hidden_size → 128 → num_domains)

---

## ⚙️ Training Configuration

### Hyperparameters (Defaults)
- **Learning Rate (PLM)**: 2e-5 (pre-trained model)
- **Learning Rate (Head)**: 1e-4 (classification head)
- **Weight Decay**: 0.01
- **Batch Size**: 16
- **Max Sequence Length**: 512 tokens
- **Epochs**: 3
- **Warmup Steps**: 500
- **Gradient Accumulation**: 2 steps
- **Gradient Clipping**: 1.0 (max norm)
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup + decay

### Training Process
1. **Forward Pass**: Model predicts logits
2. **Loss Computation**: 
   - Standard: Cross-entropy
   - R-Drop: CE + KL divergence
   - Mixup: Weighted CE
   - GroupDRO: Weighted per-sample losses
   - IRM: CE + gradient penalty
   - DANN: CE + adversarial loss
3. **Backward Pass**: Gradient computation
4. **Optimization**: AdamW step with learning rate schedule
5. **Validation**: After each epoch

---

## 📈 Evaluation Metrics

The pipeline computes and reports:

1. **Accuracy**: `correct_predictions / total_predictions`
2. **Precision**: `TP / (TP + FP)` (macro-averaged)
3. **Recall**: `TP / (TP + FN)` (macro-averaged)
4. **F1 Score**: `2 * (precision * recall) / (precision + recall)` (macro-averaged)
5. **AUC-ROC**: Area under ROC curve (if binary classification)

### Model Selection
- **Best Model**: Selected based on **validation F1 score**
- **Saved**: Best model saved to `output/best_model/`
- **History**: Training history saved to `output/training_history.json`

---

## 📁 Output Files

After training completes:

1. **`output/best_model/`**
   - Saved model weights (HuggingFace format)
   - Can be loaded with `from_pretrained()`

2. **`output/training_history.json`**
   ```json
   {
     "train_loss": [0.5, 0.4, 0.3],
     "val_loss": [0.6, 0.5, 0.4],
     "val_f1": [0.75, 0.80, 0.82],
     "val_acc": [0.70, 0.75, 0.78]
   }
   ```

3. **`output/test_results.json`**
   ```json
   {
     "test_metrics": {
       "accuracy": 0.78,
       "precision": 0.76,
       "recall": 0.80,
       "f1": 0.78,
       "auc": 0.85
     },
     "best_val_f1": 0.82
   }
   ```

---

## 🔄 Complete Pipeline Flow

```
1. Load Datasets
   ├─ english_fake_news_2212.csv
   ├─ fake_news_dataset.csv
   └─ dataset-merged.csv
   ↓
2. Preprocess & Combine
   ├─ Lowercase, remove URLs
   ├─ Remove duplicates
   └─ Combine into single DataFrame
   ↓
3. Create Splits (Stratified)
   ├─ Train (70%)
   ├─ Validation (10%)
   └─ Test (20%)
   ↓
4. Initialize Tokenizer
   └─ XLM-RoBERTa or BERT tokenizer
   ↓
5. Create PyTorch Datasets
   ├─ Tokenize text (max_length=512)
   ├─ Encode labels (0/1)
   └─ Hash domains for grouping
   ↓
6. Create DataLoaders
   ├─ Batch size: 16
   ├─ Shuffle: True (train), False (val/test)
   └─ Num workers: 0
   ↓
7. Initialize Model
   ├─ Load pre-trained backbone
   ├─ Optional: Apply LoRA
   └─ Move to device (CUDA/CPU)
   ↓
8. Setup Training Components
   ├─ Optimizer (AdamW)
   ├─ Scheduler (Linear warmup)
   ├─ Optional: GroupDRO/IRM/DANN
   └─ Loss functions
   ↓
9. Training Loop (3 epochs)
   ├─ Forward pass
   ├─ Compute loss (with regularization)
   ├─ Backward pass
   ├─ Optimizer step
   └─ Validation after each epoch
   ↓
10. Save Best Model
    └─ Based on validation F1
    ↓
11. Test Evaluation
    └─ Final metrics on test set
    ↓
12. Save Results
    ├─ training_history.json
    └─ test_results.json
```

---

## 🚀 Running the Training

### Basic Command
```bash
python run_training.py
```

### With Options
```bash
python run_training.py \
  --model_name xlm-roberta-base \
  --batch_size 16 \
  --num_epochs 3 \
  --use_lora \
  --use_rdrop \
  --use_groupdro \
  --lr_plm 2e-5 \
  --lr_head 1e-4 \
  --output_dir output
```

### All Available Flags
- `--dataset_dir`: Dataset directory (default: 'dataset')
- `--model_name`: 'xlm-roberta-base' or 'bert-base-multilingual-cased'
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Training epochs (default: 3)
- `--max_length`: Max sequence length (default: 512)
- `--use_lora`: Enable LoRA fine-tuning
- `--lora_rank`: LoRA rank (default: 8)
- `--use_rdrop`: Enable R-Drop regularization
- `--use_mixup`: Enable Mixup augmentation
- `--use_groupdro`: Enable GroupDRO
- `--use_irm`: Enable IRM
- `--use_dann`: Enable DANN
- `--lr_plm`: Learning rate for pre-trained model (default: 2e-5)
- `--lr_head`: Learning rate for classifier head (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_steps`: Warmup steps (default: 500)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 2)
- `--output_dir`: Output directory (default: 'output')
- `--seed`: Random seed (default: 42)

---

## 📊 Expected Results Format

### Console Output Example
```
======================================================================
FAKE NEWS DETECTION TRAINING PIPELINE
======================================================================
======================================================================
Loading and preprocessing datasets...
======================================================================
✓ Loaded english_fake_news_2212.csv: 2212 rows
✓ Loaded fake_news_dataset.csv: 5000 rows
✓ Loaded dataset-merged.csv: 10000 rows

Total combined rows: 17212
Label distribution: Counter({0: 8606, 1: 8606})
Language distribution: Counter({'en': 15000, 'hi': 2212})

======================================================================
Creating train/val/test splits...
======================================================================
Train: 12048 (70.0%)
Val: 1721 (10.0%)
Test: 3443 (20.0%)

======================================================================
Loading tokenizer: xlm-roberta-base
======================================================================

Starting training for 3 epochs...
Using device: cuda
Regularization: R-Drop=False, Mixup=False
Domain Gen: GroupDRO=False, IRM=False, DANN=False

Epoch 1/3
Training: 100%|████████| 754/754 [05:23<00:00, loss=0.4523]
Evaluating: 100%|████████| 108/108 [00:12<00:00]
Train Loss: 0.4523
Val Loss: 0.3891
Val Acc: 0.8234
Val F1: 0.8156
Val Precision: 0.8123
Val Recall: 0.8190
Val AUC: 0.9012
✓ Saved best model with F1: 0.8156

...

Test Results:
  Accuracy: 0.8123
  F1 Score: 0.8056
  Precision: 0.7989
  Recall: 0.8123
  AUC: 0.8945
```

---

## 🔍 Key Features

1. **Multilingual Support**: Handles English and Hindi text
2. **Domain Adaptation**: Multiple techniques for cross-domain generalization
3. **Parameter Efficiency**: Optional LoRA for faster training
4. **Robust Training**: Multiple regularization techniques
5. **Stratified Splits**: Ensures balanced train/val/test distribution
6. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC
7. **Model Checkpointing**: Saves best model based on validation F1
8. **Reproducibility**: Fixed random seeds

---

## 💡 Notes

- **Device**: Automatically uses CUDA if available, else CPU
- **Memory**: Batch size 16 with gradient accumulation 2 = effective batch size 32
- **Tokenization**: Text truncated/padded to 512 tokens
- **Domain Hashing**: Domains hashed to integers (0-999) for grouping
- **Mixed Precision**: Not implemented (can be added for faster training)
- **Early Stopping**: Not implemented (saves best model instead)

