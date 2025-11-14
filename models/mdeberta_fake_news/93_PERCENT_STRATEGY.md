# Strategy to Achieve 93%+ Accuracy (NO RoBERTa)

## Current Status
- **Current Accuracy:** 82.33%
- **Target:** 93%+
- **Constraint:** No RoBERTa models

## Recommended Approach: 3-Phase Strategy

### Phase 1: Single Large Model (Target: 88-90%)
**Use:** `microsoft/mdeberta-v3-large` (560M parameters)

```bash
cd models/mdeberta_fake_news
python run_training_93percent.py --model microsoft/mdeberta-v3-large
```

**Why this works:**
- 3x larger than base model (560M vs 184M params)
- Better attention mechanisms
- LoRA fine-tuning for efficiency
- Advanced hyperparameters

**Expected Result:** 88-90% accuracy

---

### Phase 2: Ensemble Method (Target: 91-93%) ⭐ RECOMMENDED
**Train 2-3 different DeBERTa models and combine predictions**

```bash
# Option A: Use ensemble script
python ENSEMBLE_TRAINING.py

# Option B: Train manually
python run_training_93percent.py --model microsoft/mdeberta-v3-large
python run_training_93percent.py --model microsoft/mdeberta-v3-base
python run_training_93percent.py --model microsoft/deberta-v3-large
```

**Models to ensemble:**
1. `microsoft/mdeberta-v3-large` (best single model)
2. `microsoft/mdeberta-v3-base` (different architecture)
3. `microsoft/deberta-v3-large` (English-focused, very powerful)

**Ensemble prediction code:**
```python
# Average predictions from 3 models
final_pred = (pred1 + pred2 + pred3) / 3
```

**Expected Result:** 91-93% accuracy

---

### Phase 3: Advanced Techniques (Target: 93%+)
If ensemble doesn't reach 93%, add:

1. **Data Augmentation**
   - Back-translation
   - Synonym replacement
   - Paraphrasing

2. **Pseudo-labeling**
   - Use model predictions on unlabeled data
   - Retrain with confident predictions

3. **Hyperparameter Tuning**
   - Learning rate: 3e-6 to 8e-6
   - Batch size: 2-4
   - Epochs: 8-10

---

## Quick Start Guide

### Step 1: Try Large Model (Easiest)
```bash
python run_training_93percent.py --model microsoft/mdeberta-v3-large
```
**Time:** ~2-3 hours  
**Expected:** 88-90%

### Step 2: If < 90%, Use Ensemble
```bash
python ENSEMBLE_TRAINING.py
```
**Time:** ~6-9 hours (3 models)  
**Expected:** 91-93%

### Step 3: If Still < 93%, Fine-tune
- Adjust learning rate
- Add data augmentation
- Train for more epochs

---

## Model Comparison (Non-RoBERTa)

| Model | Parameters | Expected Accuracy | Training Time |
|-------|-----------|------------------|---------------|
| mdeberta-v3-base | 184M | 82-85% | 1-2 hours |
| mdeberta-v3-large | 560M | 88-90% | 2-3 hours |
| deberta-v3-large | 435M | 87-89% | 2-3 hours |
| **Ensemble (3 models)** | - | **91-93%** | 6-9 hours |

---

## Key Features of `run_training_93percent.py`

✅ **Advanced Hyperparameters:**
- Max length: 512 (full context)
- Epochs: 7 (more training)
- Learning rate: 5e-6 (optimized for large models)
- Cosine with restarts scheduler
- Gradient accumulation: 4 steps
- Early stopping: 3 patience

✅ **LoRA Fine-tuning:**
- Rank: 16 (higher for better performance)
- Alpha: 32
- Trains only ~5% of parameters
- Faster training, less memory

✅ **Enhanced Preprocessing:**
- Better text cleaning
- Duplicate removal
- Stratified train/val split

✅ **Comprehensive Metrics:**
- Accuracy, F1, Precision, Recall

---

## Expected Results Timeline

| Phase | Method | Accuracy | Time |
|-------|--------|----------|------|
| Current | mdeberta-v3-base | 82.33% | - |
| Phase 1 | mdeberta-v3-large | 88-90% | 2-3h |
| Phase 2 | Ensemble (3 models) | **91-93%** | 6-9h |
| Phase 3 | + Augmentation | 93-94% | 10-12h |

---

## Troubleshooting

### If accuracy < 88% with large model:
1. Check GPU memory (need ~16GB+)
2. Reduce batch size to 1
3. Use gradient checkpointing
4. Try without LoRA: `--no-lora`

### If ensemble < 91%:
1. Train each model with different seeds
2. Use different hyperparameters per model
3. Add more models to ensemble

### Memory Issues:
- Use `--no-lora` flag
- Reduce batch size to 1
- Use gradient checkpointing
- Train models sequentially

---

## Best Practice Workflow

1. **Start with Phase 1** (single large model)
   - Quickest way to see improvement
   - If you get 90%+, you're close!

2. **If 88-90%, proceed to Phase 2** (ensemble)
   - Most reliable path to 93%
   - Train 2-3 models
   - Average predictions

3. **If still < 93%, add Phase 3** (advanced)
   - Data augmentation
   - Hyperparameter tuning
   - Pseudo-labeling

---

## Success Criteria

✅ **93%+ Accuracy Achieved When:**
- Using mdeberta-v3-large with LoRA
- OR Using ensemble of 2-3 models
- With proper hyperparameters
- After 7+ epochs

**Most Reliable Path:** Ensemble of 3 DeBERTa models

---

## Notes

- **No RoBERTa:** All models are DeBERTa variants
- **Memory:** Large models need 16GB+ GPU memory
- **Time:** Ensemble takes longer but most reliable
- **LoRA:** Recommended for efficiency, can disable if needed

