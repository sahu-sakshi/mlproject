import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class FakeNewsDatasetLoader:
    """Load and preprocess multiple fake news datasets."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        
    def load_english_fake_news_2212(self) -> pd.DataFrame:
        """Load english_fake_news_2212.csv"""
        df = pd.read_csv(self.dataset_dir / "english_fake_news_2212.csv")
        df['text'] = df['headline'].astype(str) + " " + df['body_text'].astype(str)
        df['label'] = df['label'].map({'Real': 0, 'Fake': 1})
        df['domain'] = df['source'].fillna('unknown')
        df['language'] = 'en'
        return df[['text', 'label', 'domain', 'language']].dropna()
    
    def load_fake_news_dataset(self) -> pd.DataFrame:
        """Load fake_news_dataset.csv"""
        df = pd.read_csv(self.dataset_dir / "fake_news_dataset.csv")
        df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)
        df['label'] = df['label'].map({'real': 0, 'fake': 1})
        df['domain'] = df['source'].fillna('unknown')
        df['language'] = 'en'
        return df[['text', 'label', 'domain', 'language']].dropna()
    
    def load_dataset_merged(self) -> pd.DataFrame:
        """Load dataset-merged.csv (Hindi + English)"""
        df = pd.read_csv(self.dataset_dir / "dataset-merged.csv")
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        df['domain'] = 'merged'
        # Simple language detection based on Devanagari script
        df['language'] = df['text'].apply(
            lambda x: 'hi' if bool(re.search(r'[\u0900-\u097F]', x)) else 'en'
        )
        return df[['text', 'label', 'domain', 'language']].dropna()
    
    def load_evaluation(self) -> pd.DataFrame:
        """Load evaluation.csv (semicolon-separated)"""
        df = pd.read_csv(self.dataset_dir / "evaluation.csv", sep=';')
        df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        df['domain'] = 'evaluation'
        df['language'] = 'en'
        return df[['text', 'label', 'domain', 'language']].dropna()
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def load_all_datasets(self) -> pd.DataFrame:
        """Load and combine all datasets"""
        print("Loading datasets...")
        dfs = []
        
        try:
            df1 = self.load_english_fake_news_2212()
            df1['dataset'] = 'english_fake_news_2212'
            dfs.append(df1)
            print(f"Loaded english_fake_news_2212.csv: {len(df1)} rows")
        except Exception as e:
            print(f"Error loading english_fake_news_2212.csv: {e}")
        
        try:
            df2 = self.load_fake_news_dataset()
            df2['dataset'] = 'fake_news_dataset'
            dfs.append(df2)
            print(f"Loaded fake_news_dataset.csv: {len(df2)} rows")
        except Exception as e:
            print(f"Error loading fake_news_dataset.csv: {e}")
        
        try:
            df3 = self.load_dataset_merged()
            df3['dataset'] = 'dataset_merged'
            dfs.append(df3)
            print(f"Loaded dataset-merged.csv: {len(df3)} rows")
        except Exception as e:
            print(f"Error loading dataset-merged.csv: {e}")
        
        if not dfs:
            raise ValueError("No datasets could be loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['text'] = combined_df['text'].apply(self.preprocess_text)
        
        # Remove duplicates based on text
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        
        print(f"\nTotal combined rows: {len(combined_df)}")
        print(f"Label distribution: {Counter(combined_df['label'])}")
        print(f"Language distribution: {Counter(combined_df['language'])}")
        
        return combined_df
    
    def create_splits(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_by: List[str] = ['label', 'domain']
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits"""
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=df[stratify_by].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (test_size + val_size),
            random_state=random_state,
            stratify=temp_df[stratify_by].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        )
        
        print(f"\nSplit sizes:")
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df

