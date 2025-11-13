"""
Fine-tune microsoft/mdeberta-v3-base for multilingual fake news detection.
Supports Kannada and English datasets.
"""
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from transformers import DataCollatorWithPadding
import torch
from sklearn.metrics import accuracy_score, f1_score
import os
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# STEP 1: Load and Preprocess Datasets
# ============================================================================

def load_kannada_datasets(dataset_dir="../../dataset"):
    """Load Kannada fake and true news datasets."""
    dataset_dir = Path(dataset_dir)
    
    # Load Kannada datasets
    ka_fake = pd.read_csv(dataset_dir / "ka_fake.csv")
    ka_true = pd.read_csv(dataset_dir / "ka_true.csv")
    
    # Add labels: fake=0, true=1
    ka_fake['label'] = 0
    ka_true['label'] = 1
    
    # Combine Kannada datasets
    ka_combined = pd.concat([ka_fake, ka_true], ignore_index=True)
    
    # Extract text column (assuming 'text' or 'content' or 'headline' column exists)
    # Try common column names
    text_col = None
    for col in ['text', 'content', 'headline', 'article', 'news']:
        if col in ka_combined.columns:
            text_col = col
            break
    
    if text_col is None:
        # If no text column found, use first non-label column
        text_col = [c for c in ka_combined.columns if c != 'label'][0]
        print(f"Warning: Using '{text_col}' as text column for Kannada dataset")
    
    ka_combined = ka_combined[[text_col, 'label']].rename(columns={text_col: 'text'})
    ka_combined['text'] = ka_combined['text'].astype(str)
    ka_combined['language'] = 'kannada'
    
    print(f"Loaded Kannada dataset: {len(ka_combined)} samples")
    print(f"  Fake: {len(ka_combined[ka_combined['label']==0])}, True: {len(ka_combined[ka_combined['label']==1])}")
    
    return ka_combined

def load_english_dataset(dataset_dir="../../dataset"):
    """Load English fake news dataset."""
    dataset_dir = Path(dataset_dir)
    
    # Load English dataset
    df = pd.read_csv(dataset_dir / "english_fake_news_2212.csv")
    
    # Map labels: 'Real'/'Fake' or 'real'/'fake' to 0/1
    # Fake = 0, Real/True = 1
    if 'label' in df.columns:
        if df['label'].dtype == 'object':
            df['label'] = df['label'].map({'Fake': 0, 'Real': 1, 'fake': 0, 'real': 1})
        else:
            # If already numeric, ensure fake=0, real=1
            df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    # Extract text (headline + body_text if available)
    if 'headline' in df.columns and 'body_text' in df.columns:
        df['text'] = df['headline'].astype(str) + " " + df['body_text'].astype(str)
    elif 'text' in df.columns:
        df['text'] = df['text'].astype(str)
    elif 'headline' in df.columns:
        df['text'] = df['headline'].astype(str)
    else:
        # Use first text-like column
        text_col = [c for c in df.columns if c not in ['label', 'source', 'domain']][0]
        df['text'] = df[text_col].astype(str)
        print(f"Warning: Using '{text_col}' as text column for English dataset")
    
    df = df[['text', 'label']].copy()
    df['text'] = df['text'].astype(str)
    df['language'] = 'english'
    
    print(f"Loaded English dataset: {len(df)} samples")
    print(f"  Fake: {len(df[df['label']==0])}, True: {len(df[df['label']==1])}")
    
    return df

def combine_and_preprocess_datasets(dataset_dir="../../dataset"):
    """Combine Kannada and English datasets, shuffle, and split."""
    print("=" * 70)
    print("Loading and Preprocessing Datasets")
    print("=" * 70)
    
    # Load datasets
    ka_df = load_kannada_datasets(dataset_dir)
    en_df = load_english_dataset(dataset_dir)
    
    # Combine datasets
    combined_df = pd.concat([ka_df, en_df], ignore_index=True)
    
    # Remove empty texts
    combined_df = combined_df[combined_df['text'].str.strip() != '']
    combined_df = combined_df.dropna(subset=['text'])
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nCombined dataset: {len(combined_df)} samples")
    print(f"  Kannada: {len(combined_df[combined_df['language']=='kannada'])}, English: {len(combined_df[combined_df['language']=='english'])}")
    print(f"  Fake (0): {len(combined_df[combined_df['label']==0])}, True (1): {len(combined_df[combined_df['label']==1])}")
    
    # Split into 80% train, 20% validation
    split_idx = int(0.8 * len(combined_df))
    train_df = combined_df[:split_idx].reset_index(drop=True)
    val_df = combined_df[split_idx:].reset_index(drop=True)
    
    print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(combined_df)*100:.1f}%)")
    
    return train_df, val_df

# ============================================================================
# STEP 2: Tokenization and Dataset Creation
# ============================================================================

def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize text data."""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=max_length
    )

def create_datasets(train_df, val_df, tokenizer, max_length=256):
    """Create Hugging Face Dataset objects."""
    print("\n" + "=" * 70)
    print("Creating Datasets")
    print("=" * 70)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    # Tokenize
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=['text']  # Remove original text after tokenization
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
# STEP 3: Evaluation Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# ============================================================================
# STEP 4: Main Training Function
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("mDeBERTa-v3 Multilingual Fake News Detection")
    print("=" * 70)
    
    # Step 1: Load and preprocess datasets
    train_df, val_df = combine_and_preprocess_datasets()
    
    # Step 2: Load tokenizer and model
    print("\n" + "=" * 70)
    print("Loading Model and Tokenizer")
    print("=" * 70)
    model_name = "microsoft/mdeberta-v3-base"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification: fake (0) or true (1)
    )
    
    # Step 3: Create datasets
    train_dataset, val_dataset = create_datasets(
        train_df, val_df, tokenizer, max_length=256
    )
    
    # Step 4: Set up training arguments
    print("\n" + "=" * 70)
    print("Setting Up Training Arguments")
    print("=" * 70)
    
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,  # Use mixed precision training
        save_total_limit=2,
        report_to="none",
        seed=42
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Step 5: Initialize Trainer
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
        compute_metrics=compute_metrics
    )
    
    # Step 6: Train the model
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    trainer.train()
    
    # Step 7: Evaluate final model
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    eval_results = trainer.evaluate()
    print(f"\nValidation Results:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"  Loss: {eval_results['eval_loss']:.4f}")
    
    # Step 8: Save model and tokenizer
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    output_dir = "mdeberta_fakenews_model"
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to: {output_dir}/")
    
    # Step 9: Test inference
    print("\n" + "=" * 70)
    print("Test Inference")
    print("=" * 70)
    
    # Load saved model for inference
    model_for_inference = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer_for_inference = AutoTokenizer.from_pretrained(output_dir)
    
    # Example headlines
    test_examples = [
        "Breaking: Scientists discover new planet with potential for life",
        "Shocking: Celebrity caught in scandal - you won't believe what happened!"
    ]
    
    print("\nTesting with example headlines:")
    for i, example in enumerate(test_examples, 1):
        # Tokenize
        inputs = tokenizer_for_inference(
            example,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model_for_inference(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_label].item()
        
        label_name = "TRUE" if predicted_label == 1 else "FAKE"
        print(f"\nExample {i}:")
        print(f"  Text: {example}")
        print(f"  Prediction: {label_name} (confidence: {confidence:.4f})")
        print(f"  Probabilities: FAKE={predictions[0][0]:.4f}, TRUE={predictions[0][1]:.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

