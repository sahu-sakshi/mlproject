#!/bin/bash

# Example 1: Basic training with XLM-RoBERTa
python main.py \
    --model_name xlm-roberta-base \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir output/xlmr_basic

# Example 2: Training with LoRA (parameter-efficient)
python main.py \
    --model_name xlm-roberta-base \
    --use_lora \
    --lora_rank 8 \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir output/xlmr_lora

# Example 3: Training with R-Drop regularization
python main.py \
    --model_name xlm-roberta-base \
    --use_rdrop \
    --rdrop_alpha 2.0 \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir output/xlmr_rdrop

# Example 4: Training with GroupDRO (domain generalization)
python main.py \
    --model_name xlm-roberta-base \
    --use_groupdro \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir output/xlmr_groupdro

# Example 5: Training with mBERT
python main.py \
    --model_name bert-base-multilingual-cased \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir output/mbert_basic

# Example 6: Full pipeline with LoRA + R-Drop + GroupDRO
python main.py \
    --model_name xlm-roberta-base \
    --use_lora \
    --lora_rank 16 \
    --use_rdrop \
    --rdrop_alpha 2.0 \
    --use_groupdro \
    --batch_size 16 \
    --num_epochs 5 \
    --lr_plm 2e-5 \
    --lr_head 1e-4 \
    --output_dir output/xlmr_full

