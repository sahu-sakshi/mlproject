"""
Complete training script for optimized mDeBERTa-v3-base fake news detection
Target: 93%+ accuracy
"""
import torch
import numpy as np
import random
import pandas as pd
import re
from pathlib import Path
from typing import Tuple, List
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
import argparse

from deberta_optimised import FakeNewsDataset, ModelTrainer

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class FakeNewsDatasetLoader:
    """Load and preprocess multiple fake news datasets."""
    
    def __init__(self, dataset_dir: str = "../../dataset"):
        self.dataset_dir = Path(dataset_dir)
    
    def load_english_fake_news_2212(self) -> pd.DataFrame:
        """Load english_fake_news_2212.csv"""
        df = pd.read_csv(self.dataset_dir / "english_fake_news_2212.csv")
        df['text'] = df['headline'].astype(str) + " " + df['body_text'].astype(str)
        df['label'] = df['label'].map({'Real': 0, 'Fake': 1})
        df['domain'] = df['source'].fillna('unknown')
        df['language'] = 'en'
        return df[['text', 'label', 'domain', 'language']].dropna()
    
    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train (2).csv and test (1).csv (semicolon-separated)"""
        train_df = pd.read_csv(self.dataset_dir / "train (2).csv", sep=';', encoding='utf-8')
        test_df = pd.read_csv(self.dataset_dir / "test (1).csv", sep=';', encoding='utf-8')
        
        # Combine title and text
        train_df['text'] = train_df['title'].astype(str) + " " + train_df['text'].astype(str)
        test_df['text'] = test_df['title'].astype(str) + " " + test_df['text'].astype(str)
        
        # Labels are already 0/1
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)
        
        train_df['domain'] = 'train_dataset'
        test_df['domain'] = 'test_dataset'
        train_df['language'] = 'en'
        test_df['language'] = 'en'
        
        return train_df[['text', 'label', 'domain', 'language']].dropna(), \
               test_df[['text', 'label', 'domain', 'language']].dropna()
    
    def load_kannada_datasets(self) -> pd.DataFrame:
        """Load ka_true.csv and ka_fake.csv"""
        ka_true = pd.read_csv(self.dataset_dir / "ka_true.csv")
        ka_fake = pd.read_csv(self.dataset_dir / "ka_fake.csv")
        
        ka_true['label'] = 0  # True = 0
        ka_fake['label'] = 1  # Fake = 1
        
        ka_true['domain'] = 'kannada'
        ka_fake['domain'] = 'kannada'
        ka_true['language'] = 'kn'
        ka_fake['language'] = 'kn'
        
        combined = pd.concat([ka_true, ka_fake], ignore_index=True)
        return combined[['text', 'label', 'domain', 'language']].dropna()
    
    def load_tamil_datasets(self) -> pd.DataFrame:
        """Load ta_true.csv and ta_fake.csv"""
        ta_true = pd.read_csv(self.dataset_dir / "ta_true.csv")
        ta_fake = pd.read_csv(self.dataset_dir / "ta_fake.csv")
        
        ta_true['label'] = 0  # True = 0
        ta_fake['label'] = 1  # Fake = 1
        
        ta_true['domain'] = 'tamil'
        ta_fake['domain'] = 'tamil'
        ta_true['language'] = 'ta'
        ta_fake['language'] = 'ta'
        
        combined = pd.concat([ta_true, ta_fake], ignore_index=True)
        return combined[['text', 'label', 'domain', 'language']].dropna()
    
    def load_malayalam_datasets(self) -> pd.DataFrame:
        """Load ma_true.csv and ma_fake.csv"""
        ma_true = pd.read_csv(self.dataset_dir / "ma_true.csv")
        ma_fake = pd.read_csv(self.dataset_dir / "ma_fake.csv")
        
        ma_true['label'] = 0  # True = 0
        ma_fake['label'] = 1  # Fake = 1
        
        ma_true['domain'] = 'malayalam'
        ma_fake['domain'] = 'malayalam'
        ma_true['language'] = 'ml'
        ma_fake['language'] = 'ml'
        
        combined = pd.concat([ma_true, ma_fake], ignore_index=True)
        return combined[['text', 'label', 'domain', 'language']].dropna()
    
    def load_labeled_datasets(self) -> pd.DataFrame:
        """Load LabeledAuthentic-7K.csv and LabeledFake-1K.csv"""
        authentic = pd.read_csv(self.dataset_dir / "LabeledAuthentic-7K.csv")
        fake = pd.read_csv(self.dataset_dir / "LabeledFake-1K.csv")
        
        # Combine headline and content
        authentic['text'] = authentic['headline'].astype(str) + " " + authentic['content'].astype(str)
        fake['text'] = fake['headline'].astype(str) + " " + fake['content'].astype(str)
        
        authentic['label'] = 0  # Authentic = 0
        fake['label'] = 1  # Fake = 1
        
        authentic['domain'] = authentic.get('domain', 'labeled_authentic').fillna('labeled_authentic')
        fake['domain'] = fake.get('domain', 'labeled_fake').fillna('labeled_fake')
        authentic['language'] = 'en'
        fake['language'] = 'en'
        
        combined = pd.concat([authentic, fake], ignore_index=True)
        return combined[['text', 'label', 'domain', 'language']].dropna()
    
    def load_authentic_48k(self) -> pd.DataFrame:
        """Load Authentic-48K.csv if it exists"""
        try:
            df = pd.read_csv(self.dataset_dir / "Authentic-48K.csv")
            # Try to infer structure
            if 'text' in df.columns:
                df['text'] = df['text'].astype(str)
            elif 'headline' in df.columns and 'content' in df.columns:
                df['text'] = df['headline'].astype(str) + " " + df['content'].astype(str)
            else:
                # Use first text column
                text_col = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower()][0]
                df['text'] = df[text_col].astype(str)
            
            df['label'] = 0  # All authentic
            df['domain'] = 'authentic_48k'
            df['language'] = 'en'
            return df[['text', 'label', 'domain', 'language']].dropna()
        except Exception as e:
            print(f"Warning: Could not load Authentic-48K.csv: {e}")
            return pd.DataFrame(columns=['text', 'label', 'domain', 'language'])
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def load_all_datasets(self) -> pd.DataFrame:
        """Load and combine all datasets"""
        print("=" * 70)
        print("Loading and preprocessing datasets...")
        print("=" * 70)
        
        dfs = []
        
        try:
            df1 = self.load_english_fake_news_2212()
            df1['dataset'] = 'english_fake_news_2212'
            dfs.append(df1)
            print(f"✓ Loaded english_fake_news_2212.csv: {len(df1)} rows")
        except Exception as e:
            print(f"✗ Error loading english_fake_news_2212.csv: {e}")
        
        try:
            train_df, test_df = self.load_train_test()
            train_df['dataset'] = 'train_2'
            test_df['dataset'] = 'test_1'
            dfs.append(train_df)
            dfs.append(test_df)
            print(f"✓ Loaded train (2).csv: {len(train_df)} rows")
            print(f"✓ Loaded test (1).csv: {len(test_df)} rows")
        except Exception as e:
            print(f"✗ Error loading train/test datasets: {e}")
        
        try:
            df3 = self.load_kannada_datasets()
            df3['dataset'] = 'kannada'
            dfs.append(df3)
            print(f"✓ Loaded Kannada datasets: {len(df3)} rows")
        except Exception as e:
            print(f"✗ Error loading Kannada datasets: {e}")
        
        try:
            df4 = self.load_tamil_datasets()
            df4['dataset'] = 'tamil'
            dfs.append(df4)
            print(f"✓ Loaded Tamil datasets: {len(df4)} rows")
        except Exception as e:
            print(f"✗ Error loading Tamil datasets: {e}")
        
        try:
            df5 = self.load_malayalam_datasets()
            df5['dataset'] = 'malayalam'
            dfs.append(df5)
            print(f"✓ Loaded Malayalam datasets: {len(df5)} rows")
        except Exception as e:
            print(f"✗ Error loading Malayalam datasets: {e}")
        
        try:
            df6 = self.load_labeled_datasets()
            df6['dataset'] = 'labeled'
            dfs.append(df6)
            print(f"✓ Loaded Labeled datasets: {len(df6)} rows")
        except Exception as e:
            print(f"✗ Error loading Labeled datasets: {e}")
        
        try:
            df7 = self.load_authentic_48k()
            if len(df7) > 0:
                df7['dataset'] = 'authentic_48k'
                dfs.append(df7)
                print(f"✓ Loaded Authentic-48K.csv: {len(df7)} rows")
        except Exception as e:
            print(f"✗ Error loading Authentic-48K.csv: {e}")
        
        if not dfs:
            raise ValueError("No datasets could be loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['text'] = combined_df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        combined_df = combined_df[combined_df['text'].str.len() > 10]
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        
        print(f"\nTotal combined rows: {len(combined_df)}")
        print(f"Label distribution: {Counter(combined_df['label'])}")
        print(f"Language distribution: {Counter(combined_df['language'])}")
        print(f"Dataset distribution: {Counter(combined_df['dataset'])}")
        
        return combined_df
    
    def create_splits(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits"""
        print("\n" + "=" * 70)
        print("Creating train/val/test splits...")
        print("=" * 70)
        
        # Create stratification key
        df['stratify_key'] = df['label'].astype(str) + '_' + df['dataset'].astype(str)
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=df['stratify_key']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (test_size + val_size),
            random_state=random_state,
            stratify=temp_df['stratify_key']
        )
        
        # Drop stratify_key
        train_df = train_df.drop(columns=['stratify_key'])
        val_df = val_df.drop(columns=['stratify_key'])
        test_df = test_df.drop(columns=['stratify_key'])
        
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized mDeBERTa-v3-base fake news detection training")
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default='../../dataset', 
                       help='Dataset directory (relative to model folder)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='microsoft/mdeberta-v3-base',
                       help='Base model to use')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA (default: True)')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    
    # Regularization arguments
    parser.add_argument('--use_rdrop', action='store_true', help='Use R-Drop (default: True)')
    parser.add_argument('--rdrop_alpha', type=float, default=2.0, help='R-Drop alpha')
    parser.add_argument('--use_mixup', action='store_true', help='Use Mixup (default: True)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    
    # Domain generalization arguments
    parser.add_argument('--use_groupdro', action='store_true', help='Use GroupDRO (default: True)')
    parser.add_argument('--use_dann', action='store_true', help='Use DANN (default: True)')
    
    # Training arguments
    parser.add_argument('--lr_plm', type=float, default=1e-5, help='Learning rate for PLM')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Learning rate for head')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
                       help='Gradient accumulation')
    parser.add_argument('--use_fp16', action='store_true', 
                       help='Use FP16 mixed precision (default: True)')
    
    args = parser.parse_args()
    
    # For optimized 93% accuracy run, enable all features by default
    # User can disable them explicitly if needed
    # Note: argparse with action='store_true' sets False by default, so we enable them here
    if not args.use_lora:
        args.use_lora = True  # Enable by default for optimized run
    if not args.use_rdrop:
        args.use_rdrop = True
    if not args.use_mixup:
        args.use_mixup = True
    if not args.use_groupdro:
        args.use_groupdro = True
    if not args.use_dann:
        args.use_dann = True
    if not args.use_fp16:
        args.use_fp16 = True
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/93_percent_run', 
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "=" * 70)
    print("OPTIMIZED mDeBERTa-v3-base FAKE NEWS DETECTION TRAINING")
    print("Target: 93%+ Accuracy")
    print("=" * 70)
    
    # Step 1: Load data
    loader = FakeNewsDatasetLoader(dataset_dir=args.dataset_dir)
    combined_df = loader.load_all_datasets()
    
    # Step 2: Create splits
    train_df, val_df, test_df = loader.create_splits(
        combined_df, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Normalize labels
    def normalize_label(x):
        x = str(x).strip().lower()
        if x in ["real", "0", "0.0", 0]:
            return 0
        if x in ["fake", "1", "1.0", 1]:
            return 1
        return int(float(x))
    
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].apply(normalize_label)
        if "domain" not in df.columns:
            df["domain"] = "unknown"
    
    # Step 3: Initialize tokenizer
    print("\n" + "=" * 70)
    print(f"Loading tokenizer: {args.model_name}")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Step 4: Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = FakeNewsDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        train_df['domain'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = FakeNewsDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        val_df['domain'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    test_dataset = FakeNewsDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        test_df['domain'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Step 5: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Step 6: Initialize trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        use_rdrop=args.use_rdrop,
        rdrop_alpha=args.rdrop_alpha,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_groupdro=args.use_groupdro,
        use_dann=args.use_dann,
        lr_plm=args.lr_plm,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_fp16=args.use_fp16
    )
    
    # Step 7: Train
    history = trainer.train(output_dir=args.output_dir)
    
    # Step 8: Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    trainer.val_loader = test_loader
    test_metrics = trainer.evaluate()
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Save test results
    test_results = {
        'test_metrics': test_metrics,
        'best_val_f1': trainer.best_val_f1
    }
    output_path = Path(args.output_dir)
    with open(output_path / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Check if target accuracy achieved
    if test_metrics['accuracy'] >= 0.93 and test_metrics['f1'] >= 0.93:
        print(f"\n{'='*70}")
        print("🎉 SUCCESS! Target accuracy of 93%+ achieved!")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("⚠️  Target accuracy not yet reached.")
        print(f"   Current Accuracy: {test_metrics['accuracy']:.4f} (target: 0.93)")
        print(f"   Current F1: {test_metrics['f1']:.4f} (target: 0.93)")
        print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed! Results saved to {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

