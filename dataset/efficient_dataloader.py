#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Efficient data loading with dynamic batching and memory optimization
"""
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Efficient text dataset with caching and lazy loading
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        cache_dir: str = None,
        preprocessing_num_workers: int = 4,
        overwrite_cache: bool = False,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Setup cache
        if cache_dir is None:
            cache_dir = self.data_path.parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file name
        cache_file = self.cache_dir / f"{self.data_path.stem}_cache.pkl"
        
        # Load or create cache
        if cache_file.exists() and not overwrite_cache:
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            logger.info("Processing and caching data...")
            self.data = self._process_data()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
    
    def _process_data(self) -> List[Dict[str, torch.Tensor]]:
        """Process raw data into tokenized format"""
        data = []
        
        # Handle directory structure for translation data
        if Path(self.data_path).is_dir():
            src_path = Path(self.data_path) / 'src-train.txt'
            tgt_path = Path(self.data_path) / 'tgt-train.txt'
            if not src_path.exists() or not tgt_path.exists():
                src_path = Path(self.data_path) / 'src-val.txt'
                tgt_path = Path(self.data_path) / 'tgt-val.txt'
        else:
            src_path = Path(self.data_path)
            tgt_path = Path(str(self.data_path).replace('src-', 'tgt-'))
        
        if not src_path.exists():
            logger.error(f"Source file not found: {src_path}")
            return data
            
        if not tgt_path.exists():
            logger.error(f"Target file not found: {tgt_path}")
            return data
            
        with open(src_path, 'r', encoding='utf-8') as f_src, \
             open(tgt_path, 'r', encoding='utf-8') as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            
        if len(src_lines) != len(tgt_lines):
            logger.warning(f"Source and target files have different lengths: {len(src_lines)} vs {len(tgt_lines)}")
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]
        
        samples = list(zip(src_lines, tgt_lines))
        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_samples = list(executor.map(self._process_sample, samples))
        
        data = [sample for sample in processed_samples if sample is not None]
        logger.info(f"Processed {len(data)} samples from {len(src_lines)} total lines")
        return data
    
    def _process_sample(self, sample: tuple) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample"""
        try:
            source_text, target_text = sample
            
            # Tokenize
            source_tokens = self.tokenizer.encode(source_text)
            target_tokens = self.tokenizer.encode(target_text)
            
            # Truncate if necessary
            source_tokens = source_tokens[:self.max_length - 2]  # Account for special tokens
            target_tokens = target_tokens[:self.max_length - 2]
            
            # Add special tokens
            source_tokens = [self.tokenizer.bos_token_id] + source_tokens + [self.tokenizer.eos_token_id]
            target_tokens = [self.tokenizer.bos_token_id] + target_tokens + [self.tokenizer.eos_token_id]
            
            return {
                'source': torch.tensor(source_tokens, dtype=torch.long),
                'target': torch.tensor(target_tokens, dtype=torch.long),
                'source_length': torch.tensor(len(source_tokens), dtype=torch.long),
                'target_length': torch.tensor(len(target_tokens), dtype=torch.long)
            }
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups sequences by length for efficiency
    """
    
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_tokens: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        indices = np.argsort(self.lengths)
        
        if self.shuffle:
            # Shuffle within buckets of similar lengths
            bucket_size = 100
            for i in range(0, len(indices), bucket_size):
                bucket = indices[i:i + bucket_size]
                np.random.shuffle(bucket)
                indices[i:i + bucket_size] = bucket
        
        batch = []
        max_len_in_batch = 0
        
        for idx in indices:
            length = self.lengths[idx]
            max_len_in_batch = max(max_len_in_batch, length)
            
            # Check if adding this sample would exceed limits
            would_exceed = False
            if self.num_tokens:
                would_exceed = (len(batch) + 1) * max_len_in_batch > self.num_tokens
            else:
                would_exceed = len(batch) + 1 > self.batch_size
            
            if would_exceed and batch:
                yield batch
                batch = [idx]
                max_len_in_batch = length
            else:
                batch.append(idx)
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class CollateFunction:
    """
    Custom collate function for padding and masking
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Dictionary with padded tensors and masks
        """
        # Extract sequences
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]
        
        # Pad sequences
        source_padded = self._pad_sequences(sources)
        target_padded = self._pad_sequences(targets)
        
        # Create masks
        source_mask = (source_padded != self.pad_token_id)
        target_mask = (target_padded != self.pad_token_id)
        
        # Create causal mask for decoder
        seq_len = target_padded.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        return {
            'source': source_padded,
            'target': target_padded,
            'source_mask': source_mask,
            'target_mask': target_mask,
            'causal_mask': causal_mask,
            'source_lengths': torch.tensor([len(s) for s in sources]),
            'target_lengths': torch.tensor([len(t) for t in targets])
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to the same length"""
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.full((len(sequences), max_len), self.pad_token_id, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            padded[i, :seq.size(0)] = seq
        
        return padded


class PrefetchDataLoader(DataLoader):
    """
    DataLoader with prefetching for better performance
    """
    
    def __init__(self, *args, prefetch_factor: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_factor = prefetch_factor


class DataModule:
    """
    Data module that handles all data loading needs
    """
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str = None,
        tokenizer=None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        pad_token_id: int = 0,
        use_dynamic_batching: bool = True,
        max_tokens_per_batch: Optional[int] = None,
    ):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pad_token_id = pad_token_id
        self.use_dynamic_batching = use_dynamic_batching
        self.max_tokens_per_batch = max_tokens_per_batch
        
        self.collate_fn = CollateFunction(pad_token_id)
    
    def setup(self):
        """Setup datasets"""
        self.train_dataset = TextDataset(
            self.train_data_path,
            self.tokenizer,
            self.max_length
        )
        
        self.val_dataset = TextDataset(
            self.val_data_path,
            self.tokenizer,
            self.max_length
        )
        
        if self.test_data_path:
            self.test_dataset = TextDataset(
                self.test_data_path,
                self.tokenizer,
                self.max_length
            )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        if self.use_dynamic_batching:
            lengths = [item['source_length'] + item['target_length'] 
                      for item in self.train_dataset]
            sampler = DynamicBatchSampler(
                lengths, self.batch_size, self.max_tokens_per_batch, shuffle=True
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Test dataloader"""
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        return None


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with streaming capabilities
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        collate_fn=None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
    def __iter__(self):
        # Use standard DataLoader with optimized settings
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
        
        for batch in dataloader:
            yield batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size