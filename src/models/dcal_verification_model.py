"""
DCAL Verification Model for Identical Twin Face Verification

This module implements the complete model architecture combining:
1. Vision Transformer backbone for patch embedding
2. DCAL encoder with SA, GLCA, and PWCA
3. Verification head for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from modules.backbone import VisionTransformerBackbone, create_vit_backbone
from modules.transformer import DCALEncoder
from modules.attention import AttentionRollout


class VerificationHead(nn.Module):
    """
    Binary classification head for face verification
    Takes features from two images and outputs similarity score
    """
    
    def __init__(self, 
                 feature_dim: int = 768,
                 hidden_dims: list = [512, 256],
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        
        # Feature fusion layers
        self.feature_combiner = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # Combine SA + GLCA features
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Similarity computation layers
        layers = []
        input_dim = feature_dim * 2  # Concatenated pair features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final verification layer
        layers.extend([
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        ])
        
        self.classifier = nn.Sequential(*layers)
        
        # Alternative distance-based verification
        self.distance_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, 
                features_a: torch.Tensor, 
                features_b: torch.Tensor,
                mode: str = "classification") -> torch.Tensor:
        """
        Forward pass for verification
        
        Args:
            features_a: Features from image A [batch_size, feature_dim * 2]
            features_b: Features from image B [batch_size, feature_dim * 2]
            mode: "classification" or "distance"
            
        Returns:
            verification_score: [batch_size, 1] similarity score (0-1)
        """
        if mode == "classification":
            # Classification-based verification
            
            # Combine SA and GLCA features for each image
            feat_a = self.feature_combiner(features_a)
            feat_b = self.feature_combiner(features_b)
            
            # Concatenate pair features
            pair_features = torch.cat([feat_a, feat_b], dim=1)
            
            # Binary classification
            verification_score = self.classifier(pair_features)
            
        elif mode == "distance":
            # Distance-based verification
            
            # Combine features
            feat_a = self.feature_combiner(features_a)
            feat_b = self.feature_combiner(features_b)
            
            # L2 normalize features
            feat_a = F.normalize(feat_a, p=2, dim=1)
            feat_b = F.normalize(feat_b, p=2, dim=1)
            
            # Compute cosine similarity
            cosine_sim = torch.sum(feat_a * feat_b, dim=1, keepdim=True)
            
            # Convert to probability using learned threshold
            verification_score = torch.sigmoid((cosine_sim - self.distance_threshold) * 10)
            
        else:
            raise ValueError(f"Unknown verification mode: {mode}")
        
        return verification_score


class DCALVerificationModel(nn.Module):
    """
    Complete DCAL model for identical twin face verification
    
    Architecture:
    Input -> ViT Backbone -> DCAL Encoder (SA + GLCA + PWCA) -> Verification Head -> Score
    """
    
    def __init__(self,
                 img_size: int = 448,
                 patch_size: int = 16,
                 d_model: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 d_ff: int = 3072,
                 r_ratio: float = 0.15,
                 dropout: float = 0.1,
                 stochastic_depth_prob: float = 0.1,
                 backbone_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 verification_hidden_dims: list = [512, 256]):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Vision Transformer Backbone
        self.backbone = create_vit_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size
        )
        
        # Override embed_dim from backbone
        self.d_model = self.backbone.embed_dim
        
        # DCAL Encoder (SA + GLCA + PWCA)
        self.dcal_encoder = DCALEncoder(
            d_model=self.d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            r_ratio=r_ratio,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob
        )
        
        # Feature projectors for SA and GLCA branches
        self.sa_projector = nn.Linear(self.d_model, self.d_model)
        self.glca_projector = nn.Linear(self.d_model, self.d_model)
        
        # Verification Head
        self.verification_head = VerificationHead(
            feature_dim=self.d_model,
            hidden_dims=verification_hidden_dims,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def extract_features(self, 
                        img: torch.Tensor, 
                        img_pair: Optional[torch.Tensor] = None, 
                        training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract features from a single image using DCAL
        
        Args:
            img: Input image [batch_size, channels, height, width]
            img_pair: Pair image for PWCA [batch_size, channels, height, width] (training only)
            training: Whether in training mode
            
        Returns:
            Dictionary containing extracted features
        """
        # Extract patch embeddings via backbone
        x = self.backbone(img)  # [batch_size, seq_len, d_model]
        
        # Process pair image if provided
        x_pair = None
        if training and img_pair is not None:
            x_pair = self.backbone(img_pair)
        
        # DCAL encoding
        encoder_outputs = self.dcal_encoder(x, x_pair, training)
        
        # Extract CLS token features
        sa_features = encoder_outputs['sa_features'][:, 0]      # [batch_size, d_model]
        glca_features = encoder_outputs['glca_features'][:, 0]  # [batch_size, d_model]
        
        # Project features
        sa_feat = self.sa_projector(sa_features)
        glca_feat = self.glca_projector(glca_features)
        
        # Combine features
        combined_features = torch.cat([sa_feat, glca_feat], dim=1)  # [batch_size, 2*d_model]
        
        # PWCA features (training only)
        pwca_features = None
        if training and encoder_outputs['pwca_features'] is not None:
            pwca_features = encoder_outputs['pwca_features'][:, 0]  # CLS token
        
        return {
            'sa_features': sa_feat,
            'glca_features': glca_feat,
            'combined_features': combined_features,
            'pwca_features': pwca_features,
            'attention_rollout': encoder_outputs['attention_rollout'],
            'glca_attention': encoder_outputs['glca_attention'],
            'raw_sa_features': sa_features,
            'raw_glca_features': glca_features
        }
    
    def forward(self, 
                img1: torch.Tensor, 
                img2: Optional[torch.Tensor] = None, 
                training: bool = True,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for verification or feature extraction
        
        Args:
            img1: First image [batch_size, channels, height, width]
            img2: Second image [batch_size, channels, height, width] (optional)
            training: Whether in training mode
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing verification results and optional features
        """
        # Extract features from first image
        # For PWCA, we use img2 as the pair during training
        features1 = self.extract_features(img1, img2 if training else None, training)
        
        if img2 is not None:
            # Extract features from second image  
            # For PWCA, we use img1 as the pair during training
            features2 = self.extract_features(img2, img1 if training else None, training)
            
            # Verification
            verification_score = self.verification_head(
                features1['combined_features'],
                features2['combined_features']
            )
            
            results = {
                'verification_score': verification_score,
                'similarity': verification_score  # Alias for compatibility
            }
            
            if return_features:
                results.update({
                    'features1': features1,
                    'features2': features2
                })
            
            return results
        else:
            # Feature extraction only
            if return_features:
                return {'features': features1}
            else:
                return {'combined_features': features1['combined_features']}
    
    def get_attention_maps(self, 
                          img: torch.Tensor,
                          layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for visualization
        
        Args:
            img: Input image [batch_size, channels, height, width]
            layer_idx: Which layer to extract attention from (-1 for last)
            
        Returns:
            Dictionary containing attention maps
        """
        features = self.extract_features(img, training=False)
        
        return {
            'attention_rollout': features['attention_rollout'],
            'glca_attention': features['glca_attention']
        }
    
    def compute_similarity(self, 
                          img1: torch.Tensor, 
                          img2: torch.Tensor,
                          mode: str = "classification") -> torch.Tensor:
        """
        Compute similarity between two images
        
        Args:
            img1: First image [batch_size, channels, height, width]
            img2: Second image [batch_size, channels, height, width]
            mode: Verification mode ("classification" or "distance")
            
        Returns:
            similarity_scores: [batch_size, 1]
        """
        self.eval()
        with torch.no_grad():
            features1 = self.extract_features(img1, training=False)
            features2 = self.extract_features(img2, training=False)
            
            similarity = self.verification_head(
                features1['combined_features'],
                features2['combined_features'],
                mode=mode
            )
        
        return similarity


def create_dcal_model(config) -> DCALVerificationModel:
    """
    Create DCAL model from configuration
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        model: DCALVerificationModel instance
    """
    model = DCALVerificationModel(
        img_size=config.INPUT_SIZE,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.SA_BLOCKS,
        d_ff=config.D_FF,
        r_ratio=config.LOCAL_QUERY_RATIO,
        dropout=config.DROPOUT,
        stochastic_depth_prob=config.STOCHASTIC_DEPTH,
        backbone_name=config.BACKBONE_NAME,
        pretrained=config.PRETRAINED,
        verification_hidden_dims=config.VERIFICATION_HIDDEN_DIMS
    )
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by component
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    encoder_params = sum(p.numel() for p in model.dcal_encoder.parameters())
    head_params = sum(p.numel() for p in model.verification_head.parameters())
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'backbone': backbone_params,
        'encoder': encoder_params,
        'verification_head': head_params
    }


def freeze_backbone(model: DCALVerificationModel, 
                   num_layers: int = 6) -> None:
    """
    Freeze backbone layers for fine-tuning
    
    Args:
        model: DCAL model
        num_layers: Number of layers to freeze (0 = freeze nothing)
    """
    if num_layers == 0:
        return
    
    # Freeze patch embedding and positional embedding
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Optionally unfreeze some encoder layers
    if hasattr(model.dcal_encoder, 'sa_blocks'):
        for i, block in enumerate(model.dcal_encoder.sa_blocks):
            if i >= num_layers:
                for param in block.parameters():
                    param.requires_grad = True


# Model factory function
def get_model_variants() -> Dict[str, callable]:
    """Get different model variant constructors"""
    
    def dcal_tiny(config):
        config.D_MODEL = 192
        config.NUM_HEADS = 3
        config.D_FF = 768
        config.BACKBONE_NAME = 'vit_tiny_patch16_224'
        return create_dcal_model(config)
    
    def dcal_small(config):
        config.D_MODEL = 384
        config.NUM_HEADS = 6
        config.D_FF = 1536
        config.BACKBONE_NAME = 'vit_small_patch16_224'
        return create_dcal_model(config)
    
    def dcal_base(config):
        # Default configuration
        return create_dcal_model(config)
    
    def dcal_large(config):
        config.D_MODEL = 1024
        config.NUM_HEADS = 16
        config.D_FF = 4096
        config.BACKBONE_NAME = 'vit_large_patch16_224'
        return create_dcal_model(config)
    
    return {
        'dcal_tiny': dcal_tiny,
        'dcal_small': dcal_small,
        'dcal_base': dcal_base,
        'dcal_large': dcal_large
    } 