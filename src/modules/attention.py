"""
Attention Mechanisms for Dual Cross-Attention Learning (DCAL)

This module implements the core attention mechanisms:
1. MultiHeadSelfAttention: Standard transformer self-attention
2. GlobalLocalCrossAttention: GLCA for local facial feature attention
3. PairWiseCrossAttention: PWCA for pair-wise regularization
4. AttentionRollout: Attention accumulation with residual connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MultiHeadSelfAttention(nn.Module):
    """
    Standard transformer self-attention with multiple heads
    Implementation follows the original Transformer architecture
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head self-attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class AttentionRollout(nn.Module):
    """
    Attention rollout computation with residual connections
    Implements the accumulated attention mechanism from the paper
    """
    
    def __init__(self, residual_factor: float = 0.5):
        super().__init__()
        self.residual_factor = residual_factor
        
    def forward(self, attention_weights_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute accumulated attention with residual connections
        
        Args:
            attention_weights_list: List of attention weights from each layer
                Each tensor: [batch_size, num_heads, seq_len, seq_len]
                
        Returns:
            rolled_attention: Accumulated attention [batch_size, seq_len, seq_len]
        """
        if not attention_weights_list:
            raise ValueError("Empty attention weights list")
            
        batch_size, num_heads, seq_len, _ = attention_weights_list[0].shape
        
        # Average across heads for each layer
        attention_matrices = []
        for attention_weights in attention_weights_list:
            # Average across heads: [batch_size, seq_len, seq_len]
            attention_avg = attention_weights.mean(dim=1)
            attention_matrices.append(attention_avg)
        
        # Initialize with identity matrix
        device = attention_matrices[0].device
        identity = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        rolled_attention = identity.clone()
        
        # Apply attention rollout: Ŝ_i = S̄_i ⊗ S̄_{i-1} ⊗ ... ⊗ S̄_1
        # where S̄ = 0.5*S + 0.5*I (residual connections)
        for attention in attention_matrices:
            attention_with_residual = (
                self.residual_factor * attention + 
                (1 - self.residual_factor) * identity
            )
            rolled_attention = torch.matmul(attention_with_residual, rolled_attention)
        
        return rolled_attention


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention (GLCA) for facial feature attention
    Implements cross-attention between local high-response regions and global features
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, r_ratio: float = 0.15, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.r_ratio = r_ratio
        
        # Separate linear projections for GLCA (no weight sharing with SA)
        self.W_q_local = nn.Linear(d_model, d_model, bias=False)
        self.W_k_global = nn.Linear(d_model, d_model, bias=False)
        self.W_v_global = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def select_top_r_queries(self, x: torch.Tensor, attention_rollout: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-R queries based on CLS token accumulated attention
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            attention_rollout: Accumulated attention [batch_size, seq_len, seq_len]
            
        Returns:
            x_local: Selected local queries [batch_size, num_local, d_model]
            top_indices: Indices of selected patches [batch_size, num_local]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get CLS token attention (first row of accumulated attention)
        cls_attention = attention_rollout[:, 0, :]  # [batch_size, seq_len]
        
        # Select top-R patches (excluding CLS token itself)
        num_local = max(1, int((seq_len - 1) * self.r_ratio))  # Exclude CLS token
        
        # Get top indices from patch tokens (skip CLS token at index 0)
        _, top_indices = torch.topk(cls_attention[:, 1:], num_local, dim=1)
        top_indices = top_indices + 1  # Adjust for CLS token offset
        
        # Extract top-R local queries
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, d_model)
        x_local = torch.gather(x, 1, top_indices_expanded)
        
        return x_local, top_indices
    
    def forward(self, x: torch.Tensor, attention_rollout: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Global-Local Cross-Attention
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            attention_rollout: Accumulated attention [batch_size, seq_len, seq_len]
            
        Returns:
            output: GLCA output [batch_size, num_local, d_model]
            local_attention: Local attention weights [batch_size, num_heads, num_local, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Select top-R local queries based on accumulated attention
        x_local, top_indices = self.select_top_r_queries(x, attention_rollout)
        num_local = x_local.shape[1]
        
        # Linear projections
        Q_local = self.W_q_local(x_local).view(batch_size, num_local, self.num_heads, self.d_k).transpose(1, 2)
        K_global = self.W_k_global(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_global = self.W_v_global(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Cross-attention: local queries attend to global key-values
        scores = torch.matmul(Q_local, K_global.transpose(-2, -1)) / self.scale
        local_attention = F.softmax(scores, dim=-1)
        local_attention = self.dropout(local_attention)
        
        # Apply attention to global values
        context = torch.matmul(local_attention, V_global)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, num_local, d_model)
        output = self.W_o(context)
        
        return output, local_attention


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention (PWCA) for twin regularization
    Uses contaminated attention with image pairs during training only
    Shares weights with the self-attention module
    """
    
    def __init__(self, self_attention_module: MultiHeadSelfAttention):
        super().__init__()
        # Share weights with SA module
        self.sa_module = self_attention_module
        self.dropout = nn.Dropout(self_attention_module.dropout.p)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Pair-Wise Cross-Attention
        
        Args:
            x1: Target image features [batch_size, seq_len, d_model]
            x2: Pair image features [batch_size, seq_len, d_model]
            
        Returns:
            output: Contaminated attention output [batch_size, seq_len, d_model]
            contaminated_attention: Attention weights [batch_size, num_heads, seq_len, 2*seq_len]
        """
        batch_size, seq_len, d_model = x1.shape
        
        # Use SA module's weights for Q, K, V projections
        Q1 = self.sa_module.W_q(x1).view(batch_size, seq_len, self.sa_module.num_heads, self.sa_module.d_k).transpose(1, 2)
        K1 = self.sa_module.W_k(x1).view(batch_size, seq_len, self.sa_module.num_heads, self.sa_module.d_k).transpose(1, 2)
        V1 = self.sa_module.W_v(x1).view(batch_size, seq_len, self.sa_module.num_heads, self.sa_module.d_k).transpose(1, 2)
        
        K2 = self.sa_module.W_k(x2).view(batch_size, seq_len, self.sa_module.num_heads, self.sa_module.d_k).transpose(1, 2)
        V2 = self.sa_module.W_v(x2).view(batch_size, seq_len, self.sa_module.num_heads, self.sa_module.d_k).transpose(1, 2)
        
        # Concatenate key-values from both images
        K_combined = torch.cat([K1, K2], dim=-2)  # [batch_size, num_heads, 2*seq_len, d_k]
        V_combined = torch.cat([V1, V2], dim=-2)  # [batch_size, num_heads, 2*seq_len, d_k]
        
        # Compute attention with contaminated key-values
        scores = torch.matmul(Q1, K_combined.transpose(-2, -1)) / self.sa_module.scale
        contaminated_attention = F.softmax(scores, dim=-1)  # Normalized over 2*seq_len
        contaminated_attention = self.dropout(contaminated_attention)
        
        # Apply contaminated attention to combined values
        context = torch.matmul(contaminated_attention, V_combined)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.sa_module.W_o(context)
        
        return output, contaminated_attention


# Utility functions for attention mechanisms
def create_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def apply_rotary_embeddings(q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings (optional enhancement)"""
    # Placeholder for rotary embeddings implementation
    # Can be added for improved positional encoding
    return q, k 