#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized attention mechanisms for transformer models
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention mechanism with optimized memory usage
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 bias: bool = True, scale_attention: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale_attention = scale_attention
        
        # Use single linear layer for Q, K, V projection for efficiency
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_projection = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, 
                value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model), defaults to query
            value: (batch_size, seq_len, d_model), defaults to query
            mask: (batch_size, seq_len, seq_len) or broadcastable
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, d_model = query.size()
        
        # Project Q, K, V in one go for efficiency
        qkv = self.qkv_projection(query)  # (batch_size, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.scale_attention:
            scores = scores / math.sqrt(self.d_k)
            
        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # Add head dimension
            elif mask.dim() == 3 and mask.size(1) != self.n_heads:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Final projection
        output = self.out_projection(context)
        
        return output


class EfficientSelfAttention(nn.Module):
    """
    Memory-efficient self-attention using Flash Attention principles
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(context)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for encoder-decoder attention
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.size()
        batch_size, seq_len_kv, _ = key.size()
        
        # Compute Q, K, V
        Q = self.query(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(key).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        return self.out(context)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better positional understanding
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[1]
            
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos, sin = emb.cos(), emb.sin()
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed