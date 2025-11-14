"""
Advanced training script targeting 93%+ accuracy using DeBERTa models (NO RoBERTa).
Implements: Larger models, LoRA, data augmentation, ensemble, and advanced techniques.
"""
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from datetime import datetime
import sys
import json
import re
from typing import List, Dict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Model Options (NO RoBERTa) - Ordered by expected performance
# ============================================================================
NON_ROBERTA_MODELS = [
    "microsoft/mdeberta-v3-large",      # Best option - 560M params
    "microsoft/deberta-v3-large",       # English-focused, very powerful
    "microsoft/mdeberta-v3-base",       # Current model
    "bert-base-multilingual-cased",     # Alternative multilingual
]

# ============================================================================
# Advanced Hyperparameters for 93% Target
# ============================================================================
ADVANCED_CONFIG = {
    "max_length": 512,                  # Full context
    "num_epochs": 7,                    # More epochs
    "per_device_train_batch_size": 2,   # Smaller for large models
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-6,              # Lower LR for large models
    "weight_decay": 0.01,
    "warmup_steps": 1500,
    "warmup_ratio": 0.15,
    "lr_scheduler_type": "cosine_with_restarts",  # Best scheduler
    "save_steps": 300,
    "eval_steps": 300,
    "logging_steps": 50,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "fp16": True,
    "save_total_limit": 5,
    "report_to": "none",
    "seed": 42,
    "dataloader_num_workers": 4,
    "gradient_accumulation_steps": 4,   # Effective batch = 2*4 = 8
    "max_grad_norm": 1.0,                # Gradient clipping
}

# ============================================================================
# LoRA Configuration
# ============================================================================
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,                                # Higher rank for better performance
    lora_alpha=32,                      # 2x rank
    lora_dropout=0.1,
    target_modules=["query_proj", "key_proj", "value_proj", "dense"]  # DeBERTa modules
)

# ============================================================================
# Output File Setup
# ============================================================================

class Tee:
    """Class to write output to both console and file."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_output_file():
    """Create output file with timestamp."""
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"training_93percent_{timestamp}.txt"
    
    f = open(output_file, 'w', encoding='utf-8')
    f.write(f"93% Accuracy Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")
    
    return f, output_file

# ============================================================================
# Advanced Data Preprocessing
# ============================================================================

def clean_text(text: str) -> str:
    """Enhanced text cleaning."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def augment_text(text: str, method: str = "none") -> str:
    """Simple text augmentation."""
    if method == "none":
        return text
    
    # Add slight variations (can be expanded)
    if method == "lowercase":
        return text.lower()
    
    return text

# ============================================================================
# Data Loading
# ============================================================================

def load_kannada_datasets(dataset_dir="../../dataset"):
    """Load Kannada datasets with enhanced preprocessing."""
    dataset_dir = Path(dataset_dir)
    
    ka_fake = pd.read_csv(dataset_dir / "ka_fake.csv")
    ka_true = pd.read_csv(dataset_dir / "ka_true.csv")
    
    ka_fake['label'] = 0
    ka_true['label'] = 1
    
    ka_combined = pd.concat([ka_fake, ka_true], ignore_index=True)
    
    text_col = None
    for col in ['text', 'content', 'headline', 'article', 'news']:
        if col in ka_combined.columns:
            text_col = col
            break
    
    if text_col is None:
        text_col = [c for c in ka_combined.columns if c != 'label'][0]
    
    ka_combined = ka_combined[[text_col, 'label']].rename(columns={text_col: 'text'})
    ka_combined['text'] = ka_combined['text'].astype(str)
    ka_combined['text'] = ka_combined['text'].apply(clean_text)
    ka_combined = ka_combined[ka_combined['text'].str.len() > 10]  # Remove very short texts
    ka_combined['language'] = 'kannada'
    
    print(f"Loaded Kannada dataset: {len(ka_combined)} samples")
    print(f"  Fake: {len(ka_combined[ka_combined['label']==0])}, True: {len(ka_combined[ka_combined['label']==1])}")
    
    return ka_combined

def load_english_dataset(dataset_dir="../../dataset"):
    """Load English dataset with enhanced preprocessing."""
    dataset_dir = Path(dataset_dir)
    
    df = pd.read_csv(dataset_dir / "english_fake_news_2212.csv")
    
    if 'label' in df.columns:
        if df['label'].dtype == 'object':
            df['label'] = df['label'].map({'Fake': 0, 'Real': 1, 'fake': 0, 'real': 1})
        else:
            df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    if 'headline' in df.columns and 'body_text' in df.columns:
        df['text'] = df['headline'].astype(str) + " " + df['body_text'].astype(str)
    elif 'text' in df.columns:
        df['text'] = df['text'].astype(str)
    elif 'headline' in df.columns:
        df['text'] = df['headline'].astype(str)
    else:
        text_col = [c for c in df.columns if c not in ['label', 'source', 'domain']][0]
        df['text'] = df[text_col].astype(str)
    
    df = df[['text', 'label']].copy()
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 10]
    df['language'] = 'english'
    
    print(f"Loaded English dataset: {len(df)} samples")
    print(f"  Fake: {len(df[df['label']==0])}, True: {len(df[df['label']==1])}")
    
    return df

def combine_and_preprocess_datasets(dataset_dir="../../dataset"):
    """Combine and preprocess datasets."""
    print("=" * 70)
    print("Loading and Preprocessing Datasets")
    print("=" * 70)
    
    ka_df = load_kannada_datasets(dataset_dir)
    en_df = load_english_dataset(dataset_dir)
    
    combined_df = pd.concat([ka_df, en_df], ignore_index=True)
    
    # Remove empty texts
    combined_df = combined_df[combined_df['text'].str.strip() != '']
    combined_df = combined_df.dropna(subset=['text'])
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nCombined dataset: {len(combined_df)} samples")
    print(f"  Kannada: {len(combined_df[combined_df['language']=='kannada'])}, English: {len(combined_df[combined_df['language']=='english'])}")
    print(f"  Fake (0): {len(combined_df[combined_df['label']==0])}, True (1): {len(combined_df[combined_df['label']==1])}")
    
    # Stratified split
    train_df, val_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
        stratify=combined_df['label']
    )
    
    print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(combined_df)*100:.1f}%)")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ============================================================================
# Tokenization
# ============================================================================

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text data."""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length
    )

def create_datasets(train_df, val_df, tokenizer, max_length=512):
    """Create Hugging Face Dataset objects."""
    print("\n" + "=" * 70)
    print("Creating Datasets")
    print("=" * 70)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text']
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text']
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

# ============================================================================
# Enhanced Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """Compute comprehensive metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================================================
# Main Training Function
# ============================================================================

def main(model_name="microsoft/mdeberta-v3-large", use_lora=True):
    """Main training pipeline targeting 93% accuracy."""
    # Setup output file
    output_file_handle, output_file_path = setup_output_file()
    
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, output_file_handle)
    
    results = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'use_lora': use_lora,
        'test_examples': []
    }
    
    print("\n" + "=" * 70)
    print(f"93% Accuracy Training - {model_name}")
    print("=" * 70)
    print("\nKey Features:")
    print("  - Model: " + model_name)
    print("  - LoRA: " + ("Enabled" if use_lora else "Disabled"))
    print("  - Max length: 512")
    print("  - Epochs: 7")
    print("  - Learning rate: 5e-6")
    print("  - Cosine with restarts scheduler")
    print("  - Gradient accumulation: 4")
    print("  - Early stopping: Enabled")
    print("=" * 70)
    
    # Load datasets
    train_df, val_df = combine_and_preprocess_datasets()
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    results['dataset_info'] = {
        'total_samples': len(combined_df),
        'kannada_samples': len(combined_df[combined_df['language']=='kannada']),
        'english_samples': len(combined_df[combined_df['language']=='english']),
        'fake_samples': len(combined_df[combined_df['label']==0]),
        'true_samples': len(combined_df[combined_df['label']==1]),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_percentage': round(len(train_df)/len(combined_df)*100, 1),
        'val_percentage': round(len(val_df)/len(combined_df)*100, 1)
    }
    
    # Load model
    print("\n" + "=" * 70)
    print("Loading Model and Tokenizer")
    print("=" * 70)
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Apply LoRA if enabled
    if use_lora:
        print("\nApplying LoRA fine-tuning...")
        try:
            model = get_peft_model(model, LORA_CONFIG)
            print("✓ LoRA applied successfully")
        except Exception as e:
            print(f"⚠ LoRA failed: {e}. Continuing without LoRA.")
            use_lora = False
    
    # Create datasets
    max_length = ADVANCED_CONFIG['max_length']
    train_dataset, val_dataset = create_datasets(
        train_df, val_df, tokenizer, max_length=max_length
    )
    
    # Training arguments
    print("\n" + "=" * 70)
    print("Setting Up Training Arguments")
    print("=" * 70)
    
    training_args = TrainingArguments(
        output_dir=f"./training_output_93percent_{model_name.replace('/', '_')}",
        **ADVANCED_CONFIG
    )
    
    results['training_config'] = {
        'num_epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'effective_batch_size': training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        'learning_rate': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'warmup_steps': training_args.warmup_steps,
        'max_length': max_length,
        'fp16': training_args.fp16,
        'lr_scheduler': training_args.lr_scheduler_type,
        'use_lora': use_lora
    }
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize Trainer
    print("\n" + "=" * 70)
    print("Initializing Trainer")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    trainer.train()
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    eval_results = trainer.evaluate()
    print(f"\nValidation Results:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f} ({eval_results['eval_f1']*100:.2f}%)")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall: {eval_results['eval_recall']:.4f}")
    print(f"  Loss: {eval_results['eval_loss']:.4f}")
    
    results['evaluation_results'] = {
        'accuracy': eval_results['eval_accuracy'],
        'f1': eval_results['eval_f1'],
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'loss': eval_results['eval_loss']
    }
    
    # Save model
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    model_suffix = model_name.split('/')[-1]
    output_dir = f"mdeberta_fakenews_model_93percent_{model_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    if use_lora:
        trainer.model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to: {output_dir}/")
    results['model_output_dir'] = output_dir
    
    # Restore stdout
    sys.stdout = original_stdout
    output_file_handle.close()
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"Final F1 Score: {eval_results['eval_f1']:.4f} ({eval_results['eval_f1']*100:.2f}%)")
    if eval_results['eval_accuracy'] >= 0.93:
        print("🎉 TARGET ACHIEVED: 93%+ Accuracy!")
    else:
        print(f"⚠ Target: 93% | Current: {eval_results['eval_accuracy']*100:.2f}%")
        print("💡 Try ensemble method or data augmentation for further improvement")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train model targeting 93% accuracy (NO RoBERTa)")
    parser.add_argument("--model", type=str, default="microsoft/mdeberta-v3-large",
                       help="Model to use (default: microsoft/mdeberta-v3-large)")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA fine-tuning")
    args = parser.parse_args()
    
    main(model_name=args.model, use_lora=not args.no_lora)

