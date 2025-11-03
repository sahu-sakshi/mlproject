"""
Complete training script that does everything:
- Loads and preprocesses all datasets
- Creates train/val/test splits
- Initializes and trains the model
- Evaluates and saves results
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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import json
import argparse
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class FakeNewsDatasetLoader:
    """Load and preprocess multiple fake news datasets."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
    
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
        df['language'] = df['text'].apply(
            lambda x: 'hi' if bool(re.search(r'[\u0900-\u097F]', x)) else 'en'
        )
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
            df2 = self.load_fake_news_dataset()
            df2['dataset'] = 'fake_news_dataset'
            dfs.append(df2)
            print(f"✓ Loaded fake_news_dataset.csv: {len(df2)} rows")
        except Exception as e:
            print(f"✗ Error loading fake_news_dataset.csv: {e}")
        
        try:
            df3 = self.load_dataset_merged()
            df3['dataset'] = 'dataset_merged'
            dfs.append(df3)
            print(f"✓ Loaded dataset-merged.csv: {len(df3)} rows")
        except Exception as e:
            print(f"✗ Error loading dataset-merged.csv: {e}")
        
        if not dfs:
            raise ValueError("No datasets could be loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['text'] = combined_df['text'].apply(self.preprocess_text)
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
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits"""
        print("\n" + "=" * 70)
        print("Creating train/val/test splits...")
        print("=" * 70)
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=df[['label', 'domain']].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (test_size + val_size),
            random_state=random_state,
            stratify=temp_df[['label', 'domain']].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        )
        
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df

# ============================================================================
# DATASET CLASSES
# ============================================================================

class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news classification"""
    def __init__(self, texts, labels, domains=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.domains = domains if domains is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        domain = int(hash(str(self.domains[idx])) % 1000) if self.domains is not None else 0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'domains': torch.tensor(domain, dtype=torch.long)
        }

# ============================================================================
# MODEL CLASSES
# ============================================================================

class FakeNewsModel(nn.Module):
    """Fake news detection model with optional LoRA"""
    def __init__(self, model_name: str, num_labels: int = 2, use_lora: bool = False, lora_rank: int = 8):
        super().__init__()
        
        if 'xlm-roberta' in model_name.lower():
            self.backbone = XLMRobertaForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        elif 'bert' in model_name.lower():
            self.backbone = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value", "key_proj", "dense"] if 'roberta' in model_name.lower() 
                             else ["query", "value", "key", "dense"]
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            print(f"Using LoRA with rank={lora_rank}")
        
        self.model_name = model_name
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# ============================================================================
# DOMAIN GENERALIZATION TECHNIQUES
# ============================================================================

class GroupDRO:
    """Group Distributionally Robust Optimization"""
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.group_losses = {}
        self.group_counts = {}
    
    def update_and_loss(self, per_sample_losses, domains):
        """Update group statistics and compute group-weighted loss"""
        for i, domain in enumerate(domains.cpu().numpy()):
            if domain not in self.group_losses:
                self.group_losses[domain] = []
                self.group_counts[domain] = 0
            self.group_losses[domain].append(per_sample_losses[i].item())
            self.group_counts[domain] += 1
        
        group_avg_losses = {}
        for domain in self.group_losses:
            if len(self.group_losses[domain]) > 0:
                domain_samples = [x for x in domains.cpu().numpy() if x == domain]
                if domain_samples:
                    group_avg_losses[domain] = sum(self.group_losses[domain][-len(domain_samples):]) / len(domain_samples)
        
        if group_avg_losses:
            group_weights = {}
            for domain in group_avg_losses:
                group_weights[domain] = np.exp(self.eta * group_avg_losses[domain])
            
            total_weight = sum(group_weights.values())
            if total_weight > 0:
                for domain in group_weights:
                    group_weights[domain] /= total_weight
            
            weighted_loss = 0.0
            for i, domain in enumerate(domains.cpu().numpy()):
                weight = group_weights.get(domain, 1.0)
                weighted_loss += per_sample_losses[i] * weight
            
            return weighted_loss / len(domains)
        else:
            return per_sample_losses.mean()

class IRMPenalty:
    """Invariant Risk Minimization penalty"""
    def __init__(self, penalty_weight: float = 1e3):
        self.penalty_weight = penalty_weight
    
    def compute_penalty(self, model, inputs, labels, domains):
        """Compute IRM penalty across domains"""
        unique_domains = torch.unique(domains)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=inputs['input_ids'].device)
        
        penalties = []
        for domain in unique_domains:
            mask = (domains == domain)
            if mask.sum() < 2:
                continue
            
            domain_inputs = {
                k: v[mask] for k, v in inputs.items() if k != 'labels'
            }
            domain_labels = labels[mask]
            
            domain_logits = model(**domain_inputs).logits
            domain_loss = F.cross_entropy(domain_logits, domain_labels, reduction='mean')
            
            dummy_scale = torch.tensor(1.0, device=domain_logits.device, requires_grad=True)
            scaled_loss = dummy_scale * domain_loss
            grad = torch.autograd.grad(scaled_loss, dummy_scale, create_graph=True)[0]
            
            penalties.append(grad ** 2)
        
        if not penalties:
            return torch.tensor(0.0, device=inputs['input_ids'].device)
        
        penalty = torch.stack(penalties).mean()
        return self.penalty_weight * penalty

class DANN:
    """Domain Adversarial Neural Network"""
    def __init__(self, hidden_size: int, num_domains: int, lambda_adv: float = 1.0):
        self.lambda_adv = lambda_adv
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_domains)
        )
    
    def compute_adversarial_loss(self, features, domains):
        """Compute adversarial domain classification loss"""
        domain_logits = self.domain_classifier(features)
        domain_loss = F.cross_entropy(domain_logits, domains)
        return self.lambda_adv * domain_loss

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    """Training class with all domain generalization techniques"""
    def __init__(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_lora: bool = False,
        lora_rank: int = 8,
        use_rdrop: bool = False,
        rdrop_alpha: float = 2.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_groupdro: bool = False,
        use_irm: bool = False,
        use_dann: bool = False,
        lr_plm: float = 2e-5,
        lr_head: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 2
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_groupdro = use_groupdro
        self.use_irm = use_irm
        self.use_dann = use_dann
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        print(f"\nInitializing model: {model_name}")
        print(f"Device: {device}")
        
        self.model = FakeNewsModel(model_name, num_labels=2, use_lora=use_lora, lora_rank=lora_rank)
        self.model.to(device)
        
        if use_groupdro:
            self.groupdro = GroupDRO(eta=0.01)
            print("✓ GroupDRO enabled")
        if use_irm:
            self.irm = IRMPenalty(penalty_weight=1e3)
            print("✓ IRM enabled")
        if use_dann:
            hidden_size = self.model.backbone.config.hidden_size
            num_domains = 100
            self.dann = DANN(hidden_size, num_domains, lambda_adv=1.0)
            self.dann.to(device)
            print("✓ DANN enabled")
        
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'classifier' not in n], 
             'lr': lr_plm, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'classifier' in n], 
             'lr': lr_head, 'weight_decay': weight_decay}
        ]
        self.optimizer = AdamW(param_groups)
        
        if use_dann:
            self.dann_optimizer = AdamW(self.dann.parameters(), lr=1e-4)
        
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        self.num_epochs = num_epochs
        self.best_val_f1 = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    
    def rdrop_loss(self, logits1, logits2, labels):
        """R-Drop: KL consistency between two forward passes"""
        ce_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2
        kl_loss = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(logits2, dim=-1),
            reduction='batchmean'
        )
        return ce_loss + self.rdrop_alpha * kl_loss
    
    def mixup_data(self, input_ids, attention_mask, labels, alpha=0.2):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = input_ids.size(0)
        index = torch.randperm(batch_size).to(input_ids.device)
        
        mixed_input_ids = lam * input_ids + (1 - lam) * input_ids[index]
        mixed_attention_mask = attention_mask | attention_mask[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_input_ids, mixed_attention_mask, labels_a, labels_b, lam
    
    def mixup_criterion(self, pred, labels_a, labels_b, lam):
        """Mixup criterion"""
        return lam * F.cross_entropy(pred, labels_a) + (1 - lam) * F.cross_entropy(pred, labels_b)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            domains = batch['domains'].to(self.device)
            
            if self.use_rdrop:
                logits1 = self.model(input_ids, attention_mask).logits
                logits2 = self.model(input_ids, attention_mask).logits
                cls_loss = self.rdrop_loss(logits1, logits2, labels)
                per_sample_losses = None
            elif self.use_mixup:
                mixed_input_ids, mixed_attention_mask, labels_a, labels_b, lam = \
                    self.mixup_data(input_ids, attention_mask, labels, self.mixup_alpha)
                logits = self.model(mixed_input_ids, mixed_attention_mask).logits
                cls_loss = self.mixup_criterion(logits, labels_a, labels_b, lam)
                per_sample_losses = None
            else:
                outputs = self.model(input_ids, attention_mask, labels=labels)
                logits = outputs.logits
                if self.use_groupdro:
                    per_sample_losses = F.cross_entropy(logits, labels, reduction='none')
                cls_loss = outputs.loss
            
            if self.use_groupdro and per_sample_losses is not None:
                loss = self.groupdro.update_and_loss(per_sample_losses, domains)
            elif self.use_irm:
                irm_penalty = self.irm.compute_penalty(
                    self.model, 
                    {'input_ids': input_ids, 'attention_mask': attention_mask},
                    labels,
                    domains
                )
                loss = cls_loss + irm_penalty
            elif self.use_dann:
                features = self.model.backbone.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state[:, 0]
                
                adv_loss = self.dann.compute_adversarial_loss(features, domains)
                loss = cls_loss + adv_loss
                
                adv_loss.backward(retain_graph=True)
                self.dann_optimizer.step()
                self.dann_optimizer.zero_grad()
            else:
                loss = cls_loss
            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, output_dir: str = "output"):
        """Main training loop"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Epochs: {self.num_epochs}")
        print(f"Regularization: R-Drop={self.use_rdrop}, Mixup={self.use_mixup}")
        print(f"Domain Gen: GroupDRO={self.use_groupdro}, IRM={self.use_irm}, DANN={self.use_dann}")
        print("=" * 70)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            val_metrics = self.evaluate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.model.backbone.save_pretrained(output_path / "best_model")
                print(f"✓ Saved best model with F1: {self.best_val_f1:.4f}")
        
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Training completed! Best F1: {self.best_val_f1:.4f}")
        print(f"{'='*70}")
        
        return self.history

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete fake news detection training pipeline")
    
    # Data arguments
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                       choices=['xlm-roberta-base', 'bert-base-multilingual-cased'],
                       help='Base model to use')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    
    # Regularization arguments
    parser.add_argument('--use_rdrop', action='store_true', help='Use R-Drop')
    parser.add_argument('--rdrop_alpha', type=float, default=2.0, help='R-Drop alpha')
    parser.add_argument('--use_mixup', action='store_true', help='Use Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    
    # Domain generalization arguments
    parser.add_argument('--use_groupdro', action='store_true', help='Use GroupDRO')
    parser.add_argument('--use_irm', action='store_true', help='Use IRM')
    parser.add_argument('--use_dann', action='store_true', help='Use DANN')
    
    # Training arguments
    parser.add_argument('--lr_plm', type=float, default=2e-5, help='Learning rate for PLM')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Learning rate for head')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "=" * 70)
    print("FAKE NEWS DETECTION TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data
    loader = FakeNewsDatasetLoader(dataset_dir=args.dataset_dir)
    combined_df = loader.load_all_datasets()
    train_df, val_df, test_df = loader.create_splits(
        combined_df, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Step 2: Initialize tokenizer
    print("\n" + "=" * 70)
    print(f"Loading tokenizer: {args.model_name}")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Step 3: Create datasets
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
    
    # Step 4: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Step 5: Initialize trainer
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
        use_irm=args.use_irm,
        use_dann=args.use_dann,
        lr_plm=args.lr_plm,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Step 6: Train
    history = trainer.train(output_dir=args.output_dir)
    
    # Step 7: Evaluate on test set
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
    with open(Path(args.output_dir) / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed! Results saved to {args.output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

