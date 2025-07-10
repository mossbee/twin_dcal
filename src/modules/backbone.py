"""
Vision Transformer Backbone for Dual Cross-Attention Learning (DCAL)

This module implements the ViT backbone components:
1. PatchEmbedding: Convert images to patch embeddings  
2. PositionalEmbedding: Add positional information
3. VisionTransformerBackbone: Complete ViT backbone with CLS token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import timm
from timm.models.vision_transformer import VisionTransformer


class PatchEmbedding(nn.Module):
    """
    Convert 448x448 face images to 16x16 patch embeddings
    Creates 28x28 = 784 patches + 1 CLS token = 785 total tokens
    """
    
    def __init__(self, 
                 img_size: int = 448, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2  # 28 * 28 = 784
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            patch_embeddings: [batch_size, num_patches, embed_dim]
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure input size matches expected
        assert height == self.img_size and width == self.img_size, \
            f"Input size ({height}, {width}) doesn't match expected ({self.img_size}, {self.img_size})"
        
        # Apply patch embedding: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.projection(x)
        
        # Flatten spatial dimensions: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        # Apply normalization
        x = self.norm(x)
        
        return x


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for face patches and CLS token
    """
    
    def __init__(self, 
                 num_patches: int = 784, 
                 embed_dim: int = 768, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Learnable positional embeddings for patches + CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to patch embeddings
        
        Args:
            x: Patch embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            x_with_pos: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class CLSToken(nn.Module):
    """
    Learnable CLS token for classification
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend CLS token to patch embeddings
        
        Args:
            x: Patch embeddings [batch_size, num_patches, embed_dim]
            
        Returns:
            x_with_cls: [batch_size, num_patches + 1, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Prepend CLS token
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x


class VisionTransformerBackbone(nn.Module):
    """
    Vision Transformer backbone for face features
    Handles patch embedding, positional encoding, and CLS token
    """
    
    def __init__(self, 
                 img_size: int = 448,
                 patch_size: int = 16, 
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size, 
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # CLS token
        self.cls_token = CLSToken(embed_dim)
        
        # Positional embedding
        self.pos_embedding = PositionalEmbedding(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            features: [batch_size, num_patches + 1, embed_dim]
        """
        # Convert to patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        x = self.cls_token(x)    # [B, num_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = self.pos_embedding(x)  # [B, num_patches + 1, embed_dim]
        
        return x


def create_vit_backbone(model_name: str = 'vit_base_patch16_224', 
                       pretrained: bool = True,
                       img_size: int = 448) -> VisionTransformerBackbone:
    """
    Create ViT backbone with optional pre-trained weights
    
    Args:
        model_name: Name of the ViT model
        pretrained: Whether to load pre-trained weights
        img_size: Input image size
        
    Returns:
        backbone: VisionTransformerBackbone instance
    """
    if pretrained:
        # Load pre-trained ViT model from timm
        pretrained_model = timm.create_model(
            model_name, 
            pretrained=True, 
            img_size=img_size,
            num_classes=0  # Remove classification head
        )
        
        # Extract configuration
        embed_dim = pretrained_model.embed_dim
        patch_size = pretrained_model.patch_embed.patch_size[0]
        
        # Create our backbone
        backbone = VisionTransformerBackbone(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Transfer weights
        if hasattr(pretrained_model.patch_embed, 'proj'):
            backbone.patch_embed.projection.weight.data.copy_(
                pretrained_model.patch_embed.proj.weight.data
            )
            if pretrained_model.patch_embed.proj.bias is not None:
                backbone.patch_embed.projection.bias.data.copy_(
                    pretrained_model.patch_embed.proj.bias.data
                )
        
        # Transfer CLS token and positional embeddings
        backbone.cls_token.cls_token.data.copy_(pretrained_model.cls_token.data)
        
        # Handle positional embedding size differences
        pretrained_pos_embed = pretrained_model.pos_embed.data
        target_pos_embed = backbone.pos_embedding.pos_embedding.data
        
        if pretrained_pos_embed.shape != target_pos_embed.shape:
            # Interpolate positional embeddings if sizes don't match
            pretrained_pos_embed = interpolate_pos_embed(
                pretrained_pos_embed, 
                target_pos_embed.shape[1]
            )
        
        backbone.pos_embedding.pos_embedding.data.copy_(pretrained_pos_embed)
        
    else:
        # Create backbone from scratch
        backbone = VisionTransformerBackbone(img_size=img_size)
    
    return backbone


def interpolate_pos_embed(pos_embed: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    """
    Interpolate positional embeddings to match target sequence length
    
    Args:
        pos_embed: Original positional embeddings [1, seq_len, embed_dim]
        target_seq_len: Target sequence length
        
    Returns:
        interpolated_pos_embed: [1, target_seq_len, embed_dim]
    """
    if pos_embed.shape[1] == target_seq_len:
        return pos_embed
    
    # Separate CLS token and patch embeddings
    cls_pos_embed = pos_embed[:, :1, :]  # CLS token
    patch_pos_embed = pos_embed[:, 1:, :] # Patch embeddings
    
    # Calculate original and target grid sizes
    orig_seq_len = patch_pos_embed.shape[1]
    orig_grid_size = int(math.sqrt(orig_seq_len))
    target_grid_size = int(math.sqrt(target_seq_len - 1))  # -1 for CLS token
    
    # Reshape to 2D grid
    embed_dim = patch_pos_embed.shape[2]
    patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, embed_dim, H, W]
    
    # Interpolate
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(target_grid_size, target_grid_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # [1, H, W, embed_dim]
    patch_pos_embed = patch_pos_embed.reshape(1, target_grid_size * target_grid_size, embed_dim)
    
    # Concatenate CLS token and interpolated patch embeddings
    interpolated_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    return interpolated_pos_embed


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling for variable input sizes
    """
    
    def __init__(self, output_size: Tuple[int, int] = (28, 28)):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptively pool features to target size
        
        Args:
            x: Input features [batch_size, channels, height, width]
            
        Returns:
            pooled: [batch_size, channels, target_h, target_w]
        """
        return F.adaptive_avg_pool2d(x, self.output_size)


# Utility functions for backbone
def freeze_backbone_layers(model: nn.Module, num_layers_to_freeze: int = 6):
    """
    Freeze the first few layers of the backbone for fine-tuning
    
    Args:
        model: Model to freeze layers in
        num_layers_to_freeze: Number of layers to freeze
    """
    layer_count = 0
    for name, param in model.named_parameters():
        if 'patch_embed' in name or 'pos_embedding' in name or 'cls_token' in name:
            param.requires_grad = False
            layer_count += 1
            if layer_count >= num_layers_to_freeze:
                break


def get_backbone_info(model_name: str) -> dict:
    """
    Get information about a ViT model
    
    Args:
        model_name: Name of the ViT model
        
    Returns:
        info: Dictionary with model information
    """
    model = timm.create_model(model_name, pretrained=False)
    
    return {
        'embed_dim': model.embed_dim,
        'num_heads': model.num_heads,
        'num_layers': len(model.blocks),
        'patch_size': model.patch_embed.patch_size,
        'input_size': model.patch_embed.img_size
    } 