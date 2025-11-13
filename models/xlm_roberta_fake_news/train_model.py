import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    XLMRobertaForSequenceClassification,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Tuple
import json
from pathlib import Path

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

class GroupDRO:
    """Group Distributionally Robust Optimization"""
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.group_losses = {}
        self.group_counts = {}
    
    def update_and_loss(self, per_sample_losses, domains):
        """Update group statistics and compute group-weighted loss"""
        # Update group statistics
        for i, domain in enumerate(domains.cpu().numpy()):
            if domain not in self.group_losses:
                self.group_losses[domain] = []
                self.group_counts[domain] = 0
            self.group_losses[domain].append(per_sample_losses[i].item())
            self.group_counts[domain] += 1
        
        # Compute group average losses
        group_avg_losses = {}
        for domain in self.group_losses:
            if len(self.group_losses[domain]) > 0:
                group_avg_losses[domain] = sum(self.group_losses[domain][-len(domains):]) / len([x for x in domains.cpu().numpy() if x == domain])
        
        # Compute group weights (exponential moving average of worst groups)
        if group_avg_losses:
            max_loss = max(group_avg_losses.values())
            group_weights = {}
            for domain in group_avg_losses:
                # Weight by loss magnitude with exponential
                group_weights[domain] = np.exp(self.eta * group_avg_losses[domain])
            
            # Normalize weights
            total_weight = sum(group_weights.values())
            if total_weight > 0:
                for domain in group_weights:
                    group_weights[domain] /= total_weight
            
            # Apply weights to per-sample losses
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
            
            # Compute gradient penalty
            domain_logits = model(**domain_inputs).logits
            domain_loss = F.cross_entropy(domain_logits, domain_labels, reduction='mean')
            
            # Gradient w.r.t. a dummy classifier scale
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
        self.num_domains = num_domains
    
    def compute_adversarial_loss(self, features, domains):
        """Compute adversarial domain classification loss with gradient reversal"""
        domain_logits = self.domain_classifier(features)
        domain_loss = F.cross_entropy(domain_logits, domains)
        return self.lambda_adv * domain_loss

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
        
        # Initialize model
        self.model = FakeNewsModel(model_name, num_labels=2, use_lora=use_lora, lora_rank=lora_rank)
        self.model.to(device)
        
        # Initialize domain generalization components
        if use_groupdro:
            self.groupdro = GroupDRO(eta=0.01)
        if use_irm:
            self.irm = IRMPenalty(penalty_weight=1e3)
        if use_dann:
            hidden_size = self.model.backbone.config.hidden_size
            num_domains = 100  # Approximate
            self.dann = DANN(hidden_size, num_domains, lambda_adv=1.0)
            self.dann.to(device)
        
        # Setup optimizer
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'classifier' not in n], 
             'lr': lr_plm, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'classifier' in n], 
             'lr': lr_head, 'weight_decay': weight_decay}
        ]
        self.optimizer = AdamW(param_groups)
        
        if use_dann:
            self.dann_optimizer = AdamW(self.dann.parameters(), lr=1e-4)
        
        # Setup scheduler
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
            
            # Main classification loss
            if self.use_rdrop:
                logits1 = self.model(input_ids, attention_mask).logits
                logits2 = self.model(input_ids, attention_mask).logits
                cls_loss = self.rdrop_loss(logits1, logits2, labels)
                per_sample_losses = None  # R-Drop doesn't support per-sample
            elif self.use_mixup:
                mixed_input_ids, mixed_attention_mask, labels_a, labels_b, lam = \
                    self.mixup_data(input_ids, attention_mask, labels, self.mixup_alpha)
                logits = self.model(mixed_input_ids, mixed_attention_mask).logits
                cls_loss = self.mixup_criterion(logits, labels_a, labels_b, lam)
                per_sample_losses = None  # Mixup doesn't support per-sample
            else:
                outputs = self.model(input_ids, attention_mask, labels=labels)
                logits = outputs.logits
                # Compute per-sample losses for GroupDRO
                if self.use_groupdro:
                    per_sample_losses = F.cross_entropy(logits, labels, reduction='none')
                cls_loss = outputs.loss
            
            # Domain generalization losses
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
                # Get features before classifier
                features = self.model.backbone.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state[:, 0]  # CLS token
                
                adv_loss = self.dann.compute_adversarial_loss(features, domains)
                loss = cls_loss + adv_loss
                
                # Backward for DANN
                adv_loss.backward(retain_graph=True)
                self.dann_optimizer.step()
                self.dann_optimizer.zero_grad()
            else:
                loss = cls_loss
            
            # Gradient accumulation
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
        
        # Compute metrics
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
        
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Using device: {self.device}")
        print(f"Regularization: R-Drop={self.use_rdrop}, Mixup={self.use_mixup}")
        print(f"Domain Gen: GroupDRO={self.use_groupdro}, IRM={self.use_irm}, DANN={self.use_dann}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.model.backbone.save_pretrained(output_path / "best_model")
                print(f"Saved best model with F1: {self.best_val_f1:.4f}")
        
        # Save training history
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed! Best F1: {self.best_val_f1:.4f}")
        return self.history

