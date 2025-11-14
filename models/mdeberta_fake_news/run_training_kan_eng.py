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
from datetime import datetime
import sys
import json

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
    """Create output file with timestamp and return file handle."""
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"training_output_{timestamp}.txt"
    
    f = open(output_file, 'w', encoding='utf-8')
    f.write(f"mDeBERTa-v3 Fake News Detection - Training Output\n")
    f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")
    
    return f, output_file

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

def save_results_to_markdown(results_dict, output_path):
    """Save training results to a markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# mDeBERTa-v3 Fake News Detection - Training Results\n\n")
        f.write(f"**Training Date:** {results_dict.get('training_date', 'N/A')}\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- **Base Model:** {results_dict.get('model_name', 'N/A')}\n")
        f.write(f"- **Task:** Binary Classification (Fake/True News)\n")
        f.write(f"- **Languages:** Kannada, English\n\n")
        
        f.write("## Dataset Information\n\n")
        dataset_info = results_dict.get('dataset_info', {})
        f.write(f"- **Total Samples:** {dataset_info.get('total_samples', 'N/A')}\n")
        f.write(f"- **Kannada Samples:** {dataset_info.get('kannada_samples', 'N/A')}\n")
        f.write(f"- **English Samples:** {dataset_info.get('english_samples', 'N/A')}\n")
        f.write(f"- **Fake News Samples:** {dataset_info.get('fake_samples', 'N/A')}\n")
        f.write(f"- **True News Samples:** {dataset_info.get('true_samples', 'N/A')}\n")
        f.write(f"- **Train Set:** {dataset_info.get('train_samples', 'N/A')} ({dataset_info.get('train_percentage', 'N/A')}%)\n")
        f.write(f"- **Validation Set:** {dataset_info.get('val_samples', 'N/A')} ({dataset_info.get('val_percentage', 'N/A')}%)\n\n")
        
        f.write("## Training Configuration\n\n")
        train_config = results_dict.get('training_config', {})
        f.write(f"- **Epochs:** {train_config.get('num_epochs', 'N/A')}\n")
        f.write(f"- **Batch Size:** {train_config.get('batch_size', 'N/A')}\n")
        f.write(f"- **Learning Rate:** {train_config.get('learning_rate', 'N/A')}\n")
        f.write(f"- **Weight Decay:** {train_config.get('weight_decay', 'N/A')}\n")
        f.write(f"- **Warmup Steps:** {train_config.get('warmup_steps', 'N/A')}\n")
        f.write(f"- **Max Length:** {train_config.get('max_length', 'N/A')}\n")
        f.write(f"- **Mixed Precision (FP16):** {train_config.get('fp16', 'N/A')}\n\n")
        
        f.write("## Evaluation Results\n\n")
        eval_results = results_dict.get('evaluation_results', {})
        accuracy = eval_results.get('accuracy', 'N/A')
        f1 = eval_results.get('f1', 'N/A')
        loss = eval_results.get('loss', 'N/A')
        f.write(f"- **Accuracy:** {accuracy:.4f if isinstance(accuracy, (int, float)) else accuracy}\n")
        f.write(f"- **F1 Score:** {f1:.4f if isinstance(f1, (int, float)) else f1}\n")
        f.write(f"- **Validation Loss:** {loss:.4f if isinstance(loss, (int, float)) else loss}\n\n")
        
        f.write("## Test Inference Examples\n\n")
        test_examples = results_dict.get('test_examples', [])
        for i, example in enumerate(test_examples, 1):
            f.write(f"### Example {i}\n\n")
            f.write(f"**Text:** {example.get('text', 'N/A')}\n\n")
            f.write(f"**Prediction:** {example.get('prediction', 'N/A')}\n\n")
            confidence = example.get('confidence', 'N/A')
            prob_fake = example.get('prob_fake', 'N/A')
            prob_true = example.get('prob_true', 'N/A')
            f.write(f"**Confidence:** {confidence:.4f if isinstance(confidence, (int, float)) else confidence}\n\n")
            f.write(f"**Probabilities:**\n")
            f.write(f"- FAKE: {prob_fake:.4f if isinstance(prob_fake, (int, float)) else prob_fake}\n")
            f.write(f"- TRUE: {prob_true:.4f if isinstance(prob_true, (int, float)) else prob_true}\n\n")
        
        f.write("## Model Output\n\n")
        f.write(f"- **Model Saved To:** {results_dict.get('model_output_dir', 'N/A')}\n\n")
        
        f.write("---\n\n")
        f.write(f"*Results generated on {results_dict.get('training_date', 'N/A')}*\n")

def main():
    """Main training pipeline."""
    # Setup output file
    output_file_handle, output_file_path = setup_output_file()
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, output_file_handle)
    
    # Initialize results dictionary
    results = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': 'microsoft/mdeberta-v3-base',
        'test_examples': []
    }
    
    print("\n" + "=" * 70)
    print("mDeBERTa-v3 Multilingual Fake News Detection")
    print("=" * 70)
    
    # Step 1: Load and preprocess datasets
    train_df, val_df = combine_and_preprocess_datasets()
    
    # Store dataset info
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
    
    max_length = 256
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
    
    # Store training configuration
    results['training_config'] = {
        'num_epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'warmup_steps': training_args.warmup_steps,
        'max_length': max_length,
        'fp16': training_args.fp16
    }
    
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
    
    # Store evaluation results
    results['evaluation_results'] = {
        'accuracy': eval_results['eval_accuracy'],
        'f1': eval_results['eval_f1'],
        'loss': eval_results['eval_loss']
    }
    
    # Step 8: Save model and tokenizer
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    output_dir = "mdeberta_fakenews_model"
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to: {output_dir}/")
    
    # Store model output directory
    results['model_output_dir'] = output_dir
    
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
        
        # Store test example results
        results['test_examples'].append({
            'text': example,
            'prediction': label_name,
            'confidence': confidence,
            'prob_fake': predictions[0][0].item(),
            'prob_true': predictions[0][1].item()
        })
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Restore stdout before saving markdown
    sys.stdout = original_stdout
    
    # Save results to markdown file
    output_md_path = Path(__file__).parent / "output.md"
    save_results_to_markdown(results, output_md_path)
    print(f"\nResults saved to: {output_md_path}")
    
    # Close the output file handle
    output_file_handle.close()

if __name__ == "__main__":
    main()

