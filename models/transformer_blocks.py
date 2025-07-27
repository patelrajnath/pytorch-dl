#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized transformer blocks with improved modularity and performance
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.attention import MultiHeadAttention, CrossAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = 'swiglu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        if activation == 'swiglu':
            # SwiGLU: x * SiLU(W1x) * W2x
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_model, d_ff)
            self.w3 = nn.Linear(d_ff, d_model)
        else:
            # Standard FFN
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'swiglu':
            x1 = self.w1(x)
            x2 = self.w2(x)
            hidden = F.silu(x1) * x2
            return self.w3(self.dropout(hidden))
        else:
            hidden = F.relu(self.w1(x))
            return self.w2(self.dropout(hidden))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (more efficient than LayerNorm)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class TransformerEncoderLayer(nn.Module):
    """
    Optimized Transformer Encoder Layer
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, 
                 dropout: float = 0.1, activation: str = 'swiglu',
                 norm_type: str = 'rmsnorm'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, mask=mask)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer Decoder Layer
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = 'swiglu',
                 norm_type: str = 'rmsnorm'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, mask=tgt_mask)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(x, memory, memory, mask=src_mask)
        x = self.dropout(x)
        x = x + residual
        
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class GradientCheckpointingMixin:
    """
    Mixin for gradient checkpointing to reduce memory usage
    """
    
    def __init__(self, use_checkpointing: bool = False):
        self.use_checkpointing = use_checkpointing
        
    def checkpoint_forward(self, layer, *args, **kwargs):
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, *args, **kwargs)
        else:
            return layer(*args, **kwargs)


class TransformerEncoder(nn.Module, GradientCheckpointingMixin):
    """
    Optimized Transformer Encoder with gradient checkpointing support
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int, 
                 d_ff: int = None, dropout: float = 0.1, 
                 activation: str = 'swiglu', norm_type: str = 'rmsnorm',
                 use_checkpointing: bool = False):
        super().__init__()
        GradientCheckpointingMixin.__init__(self, use_checkpointing)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_type)
            for _ in range(num_layers)
        ])
        
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
            
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = self.checkpoint_forward(layer, x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module, GradientCheckpointingMixin):
    """
    Optimized Transformer Decoder with gradient checkpointing support
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int,
                 d_ff: int = None, dropout: float = 0.1,
                 activation: str = 'swiglu', norm_type: str = 'rmsnorm',
                 use_checkpointing: bool = False):
        super().__init__()
        GradientCheckpointingMixin.__init__(self, use_checkpointing)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_type)
            for _ in range(num_layers)
        ])
        
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
            
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = self.checkpoint_forward(layer, x, memory, src_mask, tgt_mask)
        return self.norm(x)