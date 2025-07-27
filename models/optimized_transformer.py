#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Transformer model with improved performance and modularity
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.transformer_blocks import TransformerEncoder, TransformerDecoder
from models.embeddings.token_emb import Embeddings
from models.embeddings.position_emb import PositionalEncoding


class OptimizedTransformer(nn.Module):
    """
    Complete optimized transformer model for sequence-to-sequence tasks
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        norm_type: str = 'rmsnorm',
        use_checkpointing: bool = False,
        share_embeddings: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.src_embedding = Embeddings(d_model, src_vocab_size)
        self.tgt_embedding = Embeddings(d_model, tgt_vocab_size)
        
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding.token_emb = self.src_embedding.token_emb
            
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer layers
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, n_heads, d_ff, dropout, 
            activation, norm_type, use_checkpointing
        )
        
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, n_heads, d_ff, dropout,
            activation, norm_type, use_checkpointing
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_padding_mask(self, x: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Create padding mask for attention"""
        return (x != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create look-ahead mask for decoder self-attention"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len), required for training
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        
        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src, self.pad_token_id)
            
        if tgt is not None and tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt, self.pad_token_id)
            tgt_look_ahead_mask = self.create_look_ahead_mask(
                tgt.size(1), tgt.device
            ).unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
        
        # Embeddings and positional encoding
        src_emb = self.src_embedding(src)
        src_emb = self.pos_encoding(src_emb)
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_mask.squeeze(1).squeeze(1))
        
        if tgt is None:
            # Inference mode - return encoder output
            return encoder_output
        
        # Decode
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        decoder_output = self.decoder(
            tgt_emb, encoder_output, 
            src_mask.squeeze(1).squeeze(1),
            tgt_mask.squeeze(1).squeeze(1)
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence"""
        src_mask = self.create_padding_mask(src, self.pad_token_id)
        src_emb = self.src_embedding(src)
        src_emb = self.pos_encoding(src_emb)
        return self.encoder(src_emb, src_mask.squeeze(1).squeeze(1))
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence given memory"""
        tgt_mask = self.create_padding_mask(tgt, self.pad_token_id)
        tgt_look_ahead_mask = self.create_look_ahead_mask(
            tgt.size(1), tgt.device
        ).unsqueeze(0).unsqueeze(0)
        full_tgt_mask = tgt_mask & tgt_look_ahead_mask
        
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        decoder_output = self.decoder(
            tgt_emb, memory,
            src_mask.squeeze(1).squeeze(1) if src_mask is not None else None,
            full_tgt_mask.squeeze(1).squeeze(1)
        )
        
        return self.output_projection(decoder_output)


class OptimizedTransformerClassifier(nn.Module):
    """
    Transformer encoder for classification tasks
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        norm_type: str = 'rmsnorm',
        pool_type: str = 'mean',  # 'mean', 'max', 'cls'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pool_type = pool_type
        
        # Embeddings
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # CLS token for classification
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers, d_model, n_heads, d_ff, dropout, 
            activation, norm_type, use_checkpointing=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # Embeddings and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Add CLS token if using CLS pooling
        if self.pool_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Update mask for CLS token
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Create attention mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Encode
        encoded = self.encoder(x, mask)
        
        # Pooling
        if self.pool_type == 'cls':
            pooled = encoded[:, 0]  # CLS token
        elif self.pool_type == 'mean':
            if mask is not None:
                mask_expanded = mask.squeeze(1).squeeze(1)
                pooled = (encoded * mask_expanded.unsqueeze(-1)).sum(dim=1) / mask_expanded.sum(dim=1, keepdim=True)
            else:
                pooled = encoded.mean(dim=1)
        elif self.pool_type == 'max':
            if mask is not None:
                mask_expanded = mask.squeeze(1).squeeze(1)
                encoded = encoded.masked_fill(~mask_expanded.unsqueeze(-1), -float('inf'))
            pooled = encoded.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Classification
        return self.classifier(pooled)