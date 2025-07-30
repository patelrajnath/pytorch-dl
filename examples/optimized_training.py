#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of the optimized transformer training system
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from config.model_config import Config, get_config
from models.optimized_transformer import OptimizedTransformer
from dataset.efficient_dataloader import DataModule
from training.trainer import create_trainer


def main():
    """Main training script"""
    
    # Configuration
    config = get_config('base')
    config.update(
        data={
            'train_data_path': './sample-data/translation',
            'val_data_path': './sample-data/translation',
            'test_data_path': './sample-data/translation',
            'tokenizer_name': 'gpt2',
            'max_length': 128,
        },
        training={
            'batch_size': 8,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
        },
        logging={
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'tensorboard': True,
        }
    )
    
    # Validate configuration
    if not config.validate():
        print("Configuration validation failed!")
        return
    
    # Save configuration
    config.to_yaml('training_config.yaml')
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocab size
    config.model.vocab_size = len(tokenizer)
    
    # Setup data module
    data_module = DataModule(
        train_data_path=config.data.train_data_path,
        val_data_path=config.data.val_data_path,
        test_data_path=config.data.test_data_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        pad_token_id=tokenizer.pad_token_id,
        use_dynamic_batching=config.training.use_dynamic_batching,
        max_tokens_per_batch=config.training.max_tokens_per_batch,
    )
    
    data_module.setup()
    
    # Create model
    model = OptimizedTransformer(
        src_vocab_size=config.model.vocab_size,
        tgt_vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        activation=config.model.activation,
        norm_type=config.model.norm_type,
        use_checkpointing=config.model.use_checkpointing,
        share_embeddings=config.model.share_embeddings,
        pad_token_id=config.model.pad_token_id,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        config=config,
        device=config.system.device
    )
    
    # Start training
    trainer.train(num_epochs=config.training.num_epochs)
    
    print("Training completed!")


if __name__ == "__main__":
    main()