#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralized configuration management for the optimized transformer
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: Optional[int] = None  # Defaults to 4 * d_model
    dropout: float = 0.1
    max_seq_len: int = 512
    activation: str = 'swiglu'  # 'swiglu', 'relu', 'gelu'
    norm_type: str = 'rmsnorm'  # 'rmsnorm', 'layernorm'
    use_checkpointing: bool = False
    share_embeddings: bool = False
    pad_token_id: int = 0
    vocab_size: int = 32000


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    batch_size: int = 32
    max_tokens_per_batch: Optional[int] = None
    num_epochs: int = 10
    save_every: int = 1000
    eval_every: int = 500
    patience: int = 10
    use_dynamic_batching: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: Optional[str] = None
    max_length: int = 512
    cache_dir: str = "./cache"
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    tokenizer_name: str = "gpt2"


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "AdamW"  # 'AdamW', 'Adam', 'SGD'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    name: str = "OneCycleLR"  # 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'Linear'
    warmup_steps: int = 1000
    total_steps: Optional[int] = None
    pct_start: float = 0.1
    anneal_strategy: str = 'cos'  # 'cos', 'linear'


@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: Optional[str] = None
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            scheduler=SchedulerConfig(**config_dict.get('scheduler', {})),
            system=SystemConfig(**config_dict.get('system', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'num_encoder_layers': self.model.num_encoder_layers,
                'num_decoder_layers': self.model.num_decoder_layers,
                'd_ff': self.model.d_ff,
                'dropout': self.model.dropout,
                'max_seq_len': self.model.max_seq_len,
                'activation': self.model.activation,
                'norm_type': self.model.norm_type,
                'use_checkpointing': self.model.use_checkpointing,
                'share_embeddings': self.model.share_embeddings,
                'pad_token_id': self.model.pad_token_id,
                'vocab_size': self.model.vocab_size,
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_steps': self.training.warmup_steps,
                'max_grad_norm': self.training.max_grad_norm,
                'mixed_precision': self.training.mixed_precision,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'batch_size': self.training.batch_size,
                'max_tokens_per_batch': self.training.max_tokens_per_batch,
                'num_epochs': self.training.num_epochs,
                'save_every': self.training.save_every,
                'eval_every': self.training.eval_every,
                'patience': self.training.patience,
                'use_dynamic_batching': self.training.use_dynamic_batching,
            },
            'data': {
                'train_data_path': self.data.train_data_path,
                'val_data_path': self.data.val_data_path,
                'test_data_path': self.data.test_data_path,
                'max_length': self.data.max_length,
                'cache_dir': self.data.cache_dir,
                'preprocessing_num_workers': self.data.preprocessing_num_workers,
                'overwrite_cache': self.data.overwrite_cache,
                'tokenizer_name': self.data.tokenizer_name,
            },
            'optimizer': {
                'name': self.optimizer.name,
                'learning_rate': self.optimizer.learning_rate,
                'weight_decay': self.optimizer.weight_decay,
                'betas': self.optimizer.betas,
                'eps': self.optimizer.eps,
            },
            'scheduler': {
                'name': self.scheduler.name,
                'warmup_steps': self.scheduler.warmup_steps,
                'total_steps': self.scheduler.total_steps,
                'pct_start': self.scheduler.pct_start,
                'anneal_strategy': self.scheduler.anneal_strategy,
            },
            'system': {
                'device': self.system.device,
                'num_workers': self.system.num_workers,
                'pin_memory': self.system.pin_memory,
                'prefetch_factor': self.system.prefetch_factor,
                'persistent_workers': self.system.persistent_workers,
                'distributed': self.system.distributed,
                'local_rank': self.system.local_rank,
                'world_size': self.system.world_size,
            },
            'logging': {
                'log_dir': self.logging.log_dir,
                'checkpoint_dir': self.logging.checkpoint_dir,
                'tensorboard': self.logging.tensorboard,
                'wandb': self.logging.wandb,
                'wandb_project': self.logging.wandb_project,
                'log_level': self.logging.log_level,
            }
        }
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                nested_update(getattr(self, key), value)
            else:
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        if self.model.d_model % self.model.n_heads != 0:
            errors.append("d_model must be divisible by n_heads")
        
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.model.max_seq_len <= 0:
            errors.append("max_seq_len must be positive")
        
        if self.data.train_data_path == "":
            errors.append("train_data_path is required")
        
        if self.data.val_data_path == "":
            errors.append("val_data_path is required")
        
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        
        return True


def nested_update(obj, updates):
    """Recursively update nested objects"""
    for key, value in updates.items():
        if hasattr(obj, key):
            if isinstance(value, dict):
                nested_update(getattr(obj, key), value)
            else:
                setattr(obj, key, value)


# Default configuration templates
DEFAULT_CONFIGS = {
    'small': {
        'model': {
            'd_model': 256,
            'n_heads': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'd_ff': 1024,
            'dropout': 0.1,
            'max_seq_len': 512,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
        }
    },
    'base': {
        'model': {
            'd_model': 512,
            'n_heads': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'max_seq_len': 512,
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 20,
        }
    },
    'large': {
        'model': {
            'd_model': 1024,
            'n_heads': 16,
            'num_encoder_layers': 12,
            'num_decoder_layers': 12,
            'd_ff': 4096,
            'dropout': 0.1,
            'max_seq_len': 1024,
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'num_epochs': 30,
            'use_checkpointing': True,
        }
    }
}


def get_config(config_name: str = 'base') -> Config:
    """Get predefined configuration"""
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    
    config_dict = DEFAULT_CONFIGS[config_name]
    return Config.from_dict(config_dict)


# Example usage
if __name__ == "__main__":
    # Create default config
    config = get_config('base')
    
    # Update with custom values
    config.update(
        data={'train_data_path': './data/train.txt'},
        training={'batch_size': 64}
    )
    
    # Validate
    if config.validate():
        print("Configuration is valid!")
        
        # Save config
        config.to_yaml('config.yaml')
        config.to_json('config.json')
        
        # Load config
        loaded_config = Config.from_yaml('config.yaml')
        print("Configuration loaded successfully!")