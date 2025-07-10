"""
Transformer Blocks for Dual Cross-Attention Learning (DCAL)

This module implements the transformer architecture components:
1. TransformerBlock: Standard transformer encoder block
2. GLCABlock: Block with Global-Local Cross-Attention
3. PWCABlock: Block with Pair-Wise Cross-Attention
4. DCALEncoder: Complete encoder combining SA, GLCA, and PWCA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from .attention import (
    MultiHeadSelfAttention,
    GlobalLocalCrossAttention,
    PairWiseCrossAttention,
    AttentionRollout
)


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation
    Standard transformer FFN implementation
    """
    
    def __init__(self, d_model: int = 768, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block (MSA + FFN + LN + residuals)
    Pre-norm architecture for better training stability
    """
    
    def __init__(self, 
                 d_model: int = 768, 
                 num_heads: int = 12, 
                 d_ff: int = 3072, 
                 dropout: float = 0.1,
                 stochastic_depth_prob: float = 0.0):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
        # Stochastic depth for regularization
        self.stochastic_depth_prob = stochastic_depth_prob
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass of transformer block
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Block output [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights
        """
        # Self-attention with pre-norm and residual connection
        attn_out, attn_weights = self.self_attention(self.norm1(x))
        
        # Apply stochastic depth during training
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                attn_out = attn_out * 0
        
        x = x + self.dropout(attn_out)
        
        # Feed-forward with pre-norm and residual connection
        ff_out = self.feed_forward(self.norm2(x))
        
        # Apply stochastic depth during training
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                ff_out = ff_out * 0
        
        x = x + self.dropout(ff_out)
        
        if return_attention:
            return x, attn_weights
        return x


class GLCABlock(nn.Module):
    """
    Transformer block with Global-Local Cross-Attention
    Separate weights from self-attention blocks
    """
    
    def __init__(self, 
                 d_model: int = 768, 
                 num_heads: int = 12, 
                 d_ff: int = 3072,
                 r_ratio: float = 0.15,
                 dropout: float = 0.1):
        super().__init__()
        
        self.glca = GlobalLocalCrossAttention(d_model, num_heads, r_ratio, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
        # Learnable combination weights for features
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, attention_rollout: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GLCA block
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            attention_rollout: Accumulated attention [batch_size, seq_len, seq_len]
            
        Returns:
            output: GLCA block output [batch_size, seq_len, d_model]
            local_attention: Local attention weights
        """
        # Global-Local Cross-Attention with pre-norm
        local_out, local_attention = self.glca(self.norm1(x), attention_rollout)
        
        # Get the indices that were actually used in GLCA (don't recompute)
        batch_size, seq_len, d_model = x.shape
        num_local = local_out.shape[1]
        
        # Create full-sized output by distributing local features efficiently
        full_local_out = torch.zeros_like(x)
        
        # Get the same indices that GLCA used (to ensure consistency)
        cls_attention = attention_rollout[:, 0, :]
        num_local_expected = max(1, int((seq_len - 1) * self.glca.r_ratio))
        _, top_indices = torch.topk(cls_attention[:, 1:], num_local_expected, dim=1)
        top_indices = top_indices + 1  # Adjust for CLS token
        
        # Use vectorized assignment instead of loops (much more efficient)
        # Ensure we don't exceed the actual number of local features
        actual_num_local = min(num_local, num_local_expected)
        
        if actual_num_local > 0:
            # Create batch and feature indices for vectorized assignment
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, actual_num_local)
            selected_indices = top_indices[:, :actual_num_local]
            
            # Vectorized assignment: full_local_out[batch_indices, selected_indices] = local_out[:, :actual_num_local]
            full_local_out[batch_indices, selected_indices] = local_out[:, :actual_num_local]
        
        # Residual connection with learnable weight
        x_local = x + self.alpha * self.dropout(full_local_out)
        
        # Feed-forward with pre-norm and residual connection
        ff_out = self.feed_forward(self.norm2(x_local))
        x_final = x_local + self.dropout(ff_out)
        
        return x_final, local_attention


class PWCABlock(nn.Module):
    """
    Transformer block with Pair-Wise Cross-Attention
    Shares weights with corresponding self-attention block
    """
    
    def __init__(self, sa_block: TransformerBlock):
        super().__init__()
        
        # Share weights with SA block
        self.sa_block = sa_block
        self.pwca = PairWiseCrossAttention(sa_block.self_attention)
        
        # Use the same feed-forward and normalization layers
        self.feed_forward = sa_block.feed_forward
        self.norm1 = sa_block.norm1
        self.norm2 = sa_block.norm2
        self.dropout = sa_block.dropout
        
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Forward pass of PWCA block
        
        Args:
            x1: Target image features [batch_size, seq_len, d_model]
            x2: Pair image features [batch_size, seq_len, d_model] (training only)
            training: Whether in training mode
            
        Returns:
            output: Block output [batch_size, seq_len, d_model]
        """
        if training and x2 is not None:
            # PWCA mode: contaminated attention with image pairs
            pwca_out, _ = self.pwca(self.norm1(x1), self.norm1(x2))
            x = x1 + self.dropout(pwca_out)
        else:
            # Regular SA mode (inference or no pair available)
            # BUG FIX: Check if self_attention method exists in sa_block
            if hasattr(self.sa_block, 'self_attention'):
                attn_out, _ = self.sa_block.self_attention(self.norm1(x1))
            else:
                # Fallback: use the sa_block's forward method (may not return attention)
                attn_out = self.sa_block(self.norm1(x1))
                # Handle case where sa_block forward returns tuple vs tensor
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            x = x1 + self.dropout(attn_out)
        
        # Feed-forward with pre-norm and residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class DCALEncoder(nn.Module):
    """
    Complete DCAL encoder combining SA, GLCA, and PWCA
    Implements the full dual cross-attention learning architecture
    """
    
    def __init__(self, 
                 d_model: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 d_ff: int = 3072,
                 r_ratio: float = 0.15,
                 dropout: float = 0.1,
                 stochastic_depth_prob: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # SA blocks (L=12)
        stochastic_depths = [stochastic_depth_prob * i / (num_layers - 1) for i in range(num_layers)]
        self.sa_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, stochastic_depths[i])
            for i in range(num_layers)
        ])
        
        # GLCA block (M=1) - separate weights
        self.glca_block = GLCABlock(d_model, num_heads, d_ff, r_ratio, dropout)
        
        # PWCA blocks (T=12) - share weights with SA blocks
        self.pwca_blocks = nn.ModuleList([
            PWCABlock(sa_block) for sa_block in self.sa_blocks
        ])
        
        # Attention rollout computation
        self.attention_rollout = AttentionRollout(residual_factor=0.5)
        
        # Layer normalization for final output
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                x_pair: Optional[torch.Tensor] = None, 
                training: bool = True) -> Dict[str, Any]:
        """
        Forward pass of DCAL encoder
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            x_pair: Pair image features [batch_size, seq_len, d_model] (training only)
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
                - sa_features: Self-attention features
                - glca_features: Global-local cross-attention features  
                - pwca_features: Pair-wise cross-attention features (training only)
                - attention_rollout: Accumulated attention weights
                - glca_attention: Local attention weights from GLCA
        """
        sa_attention_weights = []
        
        # Forward through SA blocks and collect attention weights
        x_sa = x
        for i, sa_block in enumerate(self.sa_blocks):
            x_sa, attn_weights = sa_block(x_sa, return_attention=True)
            sa_attention_weights.append(attn_weights)
        
        # Apply final normalization to SA features
        x_sa = self.final_norm(x_sa)
        
        # Compute attention rollout for GLCA
        attention_rollout = self.attention_rollout(sa_attention_weights)
        
        # GLCA forward pass
        x_glca, glca_attention = self.glca_block(x_sa, attention_rollout)
        x_glca = self.final_norm(x_glca)
        
        # PWCA forward pass (training only)
        x_pwca = None
        if training and x_pair is not None:
            x_pwca = x_pair
            for pwca_block in self.pwca_blocks:
                x_pwca = pwca_block(x_pwca, x, training=True)
            x_pwca = self.final_norm(x_pwca)
        
        return {
            'sa_features': x_sa,
            'glca_features': x_glca,
            'pwca_features': x_pwca,
            'attention_rollout': attention_rollout,
            'glca_attention': glca_attention,
            'sa_attention_weights': sa_attention_weights
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


# Utility functions for transformer blocks
def apply_stochastic_depth(x: torch.Tensor, prob: float, training: bool = True) -> torch.Tensor:
    """Apply stochastic depth regularization"""
    if not training or prob == 0.0:
        return x
    
    survival_prob = 1.0 - prob
    random_tensor = survival_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
    binary_tensor = torch.floor(random_tensor)
    return x / survival_prob * binary_tensor


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create padding mask for variable length sequences"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, max_len] 