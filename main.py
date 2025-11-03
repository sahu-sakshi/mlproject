"""
Main training script for multilingual fake news detection.
"""
import argparse
from data_loader import FakeNewsDatasetLoader
from train_model import ModelTrainer, FakeNewsDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train fake news detection model")
    
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
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for parameter-efficient finetuning')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    
    # Regularization arguments
    parser.add_argument('--use_rdrop', action='store_true', help='Use R-Drop regularization')
    parser.add_argument('--rdrop_alpha', type=float, default=2.0, help='R-Drop alpha')
    parser.add_argument('--use_mixup', action='store_true', help='Use Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    
    # Domain generalization arguments
    parser.add_argument('--use_groupdro', action='store_true', help='Use GroupDRO')
    parser.add_argument('--use_irm', action='store_true', help='Use IRM')
    parser.add_argument('--use_dann', action='store_true', help='Use DANN')
    
    # Training arguments
    parser.add_argument('--lr_plm', type=float, default=2e-5, help='Learning rate for PLM')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Learning rate for classifier head')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    import torch
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load data
    print("=" * 50)
    print("Loading and preprocessing data...")
    print("=" * 50)
    
    loader = FakeNewsDatasetLoader(dataset_dir=args.dataset_dir)
    combined_df = loader.load_all_datasets()
    train_df, val_df, test_df = loader.create_splits(
        combined_df, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize trainer
    print("\n" + "=" * 50)
    print("Initializing model trainer...")
    print("=" * 50)
    
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
    
    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    history = trainer.train(output_dir=args.output_dir)
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    print("=" * 50)
    
    # Temporarily replace val_loader with test_loader
    trainer.val_loader = test_loader
    test_metrics = trainer.evaluate()
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    print(f"\nTraining completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

