#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized training loop with mixed precision and gradient accumulation
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import json

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset.efficient_dataloader import DataModule
from models.optimized_transformer import OptimizedTransformer

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0
        self.start_time = time.time()
    
    def update(self, loss: float, tokens: int):
        self.total_loss += loss
        self.total_tokens += tokens
        self.num_batches += 1
    
    def get_metrics(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
        tokens_per_sec = self.total_tokens / elapsed if elapsed > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'tokens_per_sec': tokens_per_sec,
            'total_tokens': self.total_tokens,
            'elapsed_time': elapsed
        }


class OptimizedTrainer:
    """
    Optimized trainer with mixed precision, gradient accumulation, and monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: nn.Module = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        save_every: int = 1000,
        eval_every: int = 500,
        patience: int = 10,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision and device == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.save_every = save_every
        self.eval_every = eval_every
        self.patience = patience
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Create masks
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        
        # Create causal mask
        seq_len = target.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        target_mask = target_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.mixed_precision):
            logits = self.model(source, target, source_mask, target_mask)
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target[..., 1:].contiguous()
            
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        
        return {'train_loss': loss.item() * self.gradient_accumulation_steps}
    
    def validate(self) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Create masks
                source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
                target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
                
                # Create causal mask
                seq_len = target.size(1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                target_mask = target_mask & causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Forward pass
                logits = self.model(source, target, source_mask, target_mask)
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target[..., 1:].contiguous()
                
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        return {'val_loss': avg_val_loss}
    
    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        train_metrics = MetricsTracker()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_metrics.reset()
            
            # Training
            for batch_idx, batch in enumerate(self.train_dataloader):
                metrics = self.train_step(batch)
                train_metrics.update(metrics['train_loss'], batch['source'].numel())
                
                self.global_step += 1
                
                # Logging
                if self.global_step % 100 == 0:
                    current_metrics = train_metrics.get_metrics()
                    logger.info(
                        f"Step {self.global_step}: "
                        f"Loss: {current_metrics['loss']:.4f}, "
                        f"Tokens/sec: {current_metrics['tokens_per_sec']:.2f}"
                    )
                    
                    # Tensorboard logging
                    for key, value in current_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                
                # Validation
                if self.global_step % self.eval_every == 0:
                    val_metrics = self.validate()
                    logger.info(f"Validation - Loss: {val_metrics['val_loss']:.4f}")
                    
                    # Tensorboard logging
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'val/{key}', value, self.global_step)
                    
                    # Early stopping
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.patience_counter = 0
                        self.save_checkpoint('best_model.pt', is_best=True)
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        logger.info("Early stopping triggered")
                        break
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            if self.patience_counter >= self.patience:
                break
        
        self.writer.close()
        logger.info("Training completed")


class OptimizedTrainerConfig:
    """Configuration for trainer"""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        save_every: int = 1000,
        eval_every: int = 500,
        patience: int = 10,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.patience = patience
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'save_every': self.save_every,
            'eval_every': self.eval_every,
            'patience': self.patience,
        }
    
    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OptimizedTrainerConfig':
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def create_trainer(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    config: OptimizedTrainerConfig,
    device: str = 'cuda'
) -> OptimizedTrainer:
    """Factory function to create trainer"""
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    return OptimizedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **config.to_dict()
    )