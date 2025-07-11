"""
Verification Loss Functions for Twin Face Verification

This module implements specialized loss functions for face verification:
1. TripletLoss: For learning discriminative embeddings
2. FocalLoss: For handling class imbalance and hard examples
3. CombinedLoss: Multi-task loss combining verification objectives
4. Dynamic loss weighting for SA, GLCA, and PWCA branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math


class TripletLoss(nn.Module):
    """
    Triplet loss for learning discriminative face embeddings
    Uses online hard negative mining for better convergence
    """
    
    def __init__(self, 
                 margin: float = 0.3,
                 mining: str = "hard",
                 distance_metric: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance_metric = distance_metric
        
    def forward(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss with hard negative mining
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Binary labels [batch_size]
            
        Returns:
            loss: Triplet loss
            stats: Dictionary with loss statistics
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device), {}
        
        # Normalize embeddings for cosine distance
        if self.distance_metric == "cosine":
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = self._compute_distances(embeddings)
        
        # Create masks for positive and negative pairs
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = (labels != labels.t()).float()
        
        # Remove diagonal (self-comparisons)
        identity_mask = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask - identity_mask
        
        if self.mining == "hard":
            loss, stats = self._hard_triplet_loss(distances, positive_mask, negative_mask)
        else:
            loss, stats = self._all_triplet_loss(distances, positive_mask, negative_mask)
        
        return loss, stats
    
    def _compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between embeddings"""
        if self.distance_metric == "euclidean":
            # Euclidean distance
            dot_product = torch.matmul(embeddings, embeddings.t())
            squared_norm = torch.diag(dot_product)
            distances = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0)
            distances = torch.sqrt(distances + 1e-12)  # Add epsilon for numerical stability
        else:
            # Cosine distance
            distances = 1.0 - torch.matmul(embeddings, embeddings.t())
        
        return distances
    
    def _hard_triplet_loss(self, 
                          distances: torch.Tensor,
                          positive_mask: torch.Tensor,
                          negative_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute hard triplet loss with online hard mining"""
        batch_size = distances.size(0)
        
        # For each anchor, find hardest positive and hardest negative
        hardest_positive_dist = (distances * positive_mask).max(dim=1)[0]
        hardest_negative_dist = (distances + 1e6 * (1 - negative_mask)).min(dim=1)[0]
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        loss = triplet_loss.mean()
        
        # Statistics
        num_active_triplets = (triplet_loss > 0).sum().float()
        avg_positive_dist = hardest_positive_dist.mean()
        avg_negative_dist = hardest_negative_dist.mean()
        
        stats = {
            'triplet_loss': loss.item(),
            'active_triplets': num_active_triplets.item() / batch_size,
            'avg_positive_dist': avg_positive_dist.item(),
            'avg_negative_dist': avg_negative_dist.item()
        }
        
        return loss, stats
    
    def _all_triplet_loss(self,
                         distances: torch.Tensor,
                         positive_mask: torch.Tensor,
                         negative_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute triplet loss using all valid triplets"""
        # Get all valid triplets
        triplet_loss = distances.unsqueeze(2) - distances.unsqueeze(1) + self.margin
        
        # Mask for valid triplets (anchor-positive-negative)
        triplet_mask = positive_mask.unsqueeze(2) * negative_mask.unsqueeze(1)
        triplet_loss = triplet_loss * triplet_mask
        
        # Remove invalid triplets
        triplet_loss = F.relu(triplet_loss)
        
        # Count valid triplets
        num_triplets = triplet_mask.sum()
        
        if num_triplets > 0:
            loss = triplet_loss.sum() / num_triplets
            num_active = (triplet_loss > 0).sum().float()
            
            stats = {
                'triplet_loss': loss.item(),
                'active_triplets': (num_active / num_triplets).item(),
                'total_triplets': num_triplets.item()
            }
        else:
            loss = torch.tensor(0.0, device=distances.device)
            stats = {'triplet_loss': 0.0, 'active_triplets': 0.0, 'total_triplets': 0.0}
        
        return loss, stats


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance and hard examples
    Particularly useful for twin face verification where hard negatives are important
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute focal loss
        
        Args:
            predictions: Predicted logits [batch_size] (raw logits, not probabilities)
            targets: Binary targets [batch_size]
            
        Returns:
            loss: Focal loss
            stats: Dictionary with loss statistics
        """
        # Ensure targets are same shape as predictions
        targets = targets.view_as(predictions)
        
        # Compute binary cross entropy with logits (autocast-safe)
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Convert logits to probabilities for focal weight computation
        probs = torch.sigmoid(predictions)
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            loss = focal_loss.mean()
        elif self.reduction == "sum":
            loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        # Statistics
        stats = {
            'focal_loss': loss.item() if self.reduction != "none" else loss.mean().item(),
            'avg_focal_weight': focal_weight.mean().item(),
            'avg_confidence': p_t.mean().item()
        }
        
        return loss, stats


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for face verification
    Alternative to triplet loss for pair-based learning
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, 
                embeddings1: torch.Tensor,
                embeddings2: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss
        
        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
            labels: Binary labels (1 for same person, 0 for different) [batch_size]
            
        Returns:
            loss: Contrastive loss
            stats: Dictionary with loss statistics
        """
        # Compute euclidean distance
        distance = F.pairwise_distance(embeddings1, embeddings2, p=2)
        
        # Contrastive loss
        positive_loss = labels * torch.pow(distance, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = (positive_loss + negative_loss).mean()
        
        # Statistics
        avg_distance = distance.mean()
        positive_distances = distance[labels == 1]
        negative_distances = distance[labels == 0]
        
        stats = {
            'contrastive_loss': loss.item(),
            'avg_distance': avg_distance.item(),
            'avg_positive_distance': positive_distances.mean().item() if len(positive_distances) > 0 else 0.0,
            'avg_negative_distance': negative_distances.mean().item() if len(negative_distances) > 0 else 0.0
        }
        
        return loss, stats


class CenterLoss(nn.Module):
    """
    Center loss for learning discriminative features
    Helps minimize intra-class variation
    """
    
    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, 
                features: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute center loss
        
        Args:
            features: Feature embeddings [batch_size, feat_dim]
            labels: Class labels [batch_size] (for verification, this would be identity IDs)
            
        Returns:
            loss: Center loss
            stats: Dictionary with loss statistics
        """
        batch_size = features.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers.index_select(0, labels.long())
        loss = F.mse_loss(features, centers_batch)
        
        stats = {
            'center_loss': loss.item(),
            'center_variance': self.centers.var().item()
        }
        
        return loss, stats


class CombinedLoss(nn.Module):
    """
    Combined loss function for DCAL twin face verification
    Integrates multiple loss objectives with dynamic weighting
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Individual loss functions
        self.triplet_loss = TripletLoss(margin=config.TRIPLET_MARGIN)
        self.focal_loss = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        self.bce_loss = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for autocast safety
        
        # Loss weights
        self.loss_weights = config.LOSS_WEIGHTS.copy()
        
        # Dynamic weight adaptation
        self.weight_momentum = 0.9
        self.adaptive_weights = config.LOSS_WEIGHTS.copy()
        
    def forward(self,
                model_outputs: Dict[str, Any],
                targets: torch.Tensor,
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for DCAL model
        
        Args:
            model_outputs: Dictionary containing model outputs
            targets: Binary verification targets [batch_size, 1]
            epoch: Current training epoch (for adaptive weighting)
            
        Returns:
            total_loss: Combined loss
            loss_stats: Dictionary with detailed loss statistics
        """
        device = targets.device
        batch_size = targets.size(0)
        
        total_loss = torch.tensor(0.0, device=device)
        loss_stats = {}
        
        # Extract features and predictions
        verification_score = model_outputs.get('verification_score')
        features1 = model_outputs.get('features1', {})
        features2 = model_outputs.get('features2', {})
        
        # 1. Verification BCE Loss
        if verification_score is not None:
            # Ensure verification_score and targets have compatible shapes
            verification_score_squeezed = verification_score.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            targets_for_bce = targets.squeeze(-1) if targets.dim() > 1 else targets  # Ensure 1D
            
            bce_loss = self.bce_loss(verification_score_squeezed, targets_for_bce)
            total_loss += self.adaptive_weights['bce'] * bce_loss
            loss_stats['bce_loss'] = bce_loss.item()
        
        # 2. Focal Loss (for hard examples)
        if verification_score is not None:
            # Focal loss also needs matching shapes
            verification_score_squeezed = verification_score.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            targets_for_focal = targets.squeeze(-1) if targets.dim() > 1 else targets  # Ensure 1D
            
            focal_loss, focal_stats = self.focal_loss(verification_score_squeezed, targets_for_focal)
            total_loss += self.adaptive_weights['focal'] * focal_loss
            loss_stats.update({f'focal_{k}': v for k, v in focal_stats.items()})
        
        # 3. Triplet Loss (for embedding learning)
        if features1 and features2:
            # Combine features from both images
            combined_features = torch.cat([
                features1.get('combined_features', torch.empty(0, device=device)),
                features2.get('combined_features', torch.empty(0, device=device))
            ], dim=0)
            
            # Create labels for triplet loss (repeated for both images)
            triplet_labels = torch.cat([targets.squeeze(), targets.squeeze()], dim=0)
            
            if combined_features.numel() > 0:
                triplet_loss, triplet_stats = self.triplet_loss(combined_features, triplet_labels)
                total_loss += self.adaptive_weights['triplet'] * triplet_loss
                loss_stats.update({f'triplet_{k}': v for k, v in triplet_stats.items()})
        
        # 4. Multi-branch losses (SA, GLCA, PWCA)
        branch_losses = self._compute_branch_losses(model_outputs, targets)
        for branch_name, (branch_loss, branch_stats) in branch_losses.items():
            weight_key = f'{branch_name}_weight'
            if weight_key in self.adaptive_weights:
                total_loss += self.adaptive_weights[weight_key] * branch_loss
                loss_stats.update({f'{branch_name}_{k}': v for k, v in branch_stats.items()})
        
        # 5. Regularization losses
        reg_loss = self._compute_regularization_loss(model_outputs)
        if reg_loss > 0:
            total_loss += 0.01 * reg_loss  # Small weight for regularization
            loss_stats['regularization_loss'] = reg_loss.item()
        
        # Update adaptive weights
        self._update_adaptive_weights(loss_stats, epoch)
        
        # Add total loss and weights to stats
        loss_stats['total_loss'] = total_loss.item()
        loss_stats.update({f'weight_{k}': v for k, v in self.adaptive_weights.items()})
        
        return total_loss, loss_stats
    
    def _compute_branch_losses(self, 
                              model_outputs: Dict[str, Any],
                              targets: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute losses for individual DCAL branches"""
        branch_losses = {}
        device = targets.device
        
        features1 = model_outputs.get('features1', {})
        features2 = model_outputs.get('features2', {})
        
        # SA branch loss
        sa_feat1 = features1.get('sa_features')
        sa_feat2 = features2.get('sa_features')
        if sa_feat1 is not None and sa_feat2 is not None:
            sa_similarity = F.cosine_similarity(sa_feat1, sa_feat2, dim=1).unsqueeze(-1)  # Add dimension back
            # Don't normalize to [0,1] for BCEWithLogitsLoss - use raw similarity as logits
            # Ensure compatible shapes for BCE loss
            sa_similarity_squeezed = sa_similarity.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            targets_for_sa = targets.squeeze(-1) if targets.dim() > 1 else targets  # Ensure 1D
            sa_loss = self.bce_loss(sa_similarity_squeezed, targets_for_sa)
            branch_losses['sa'] = (sa_loss, {'loss': sa_loss.item()})
        
        # GLCA branch loss
        glca_feat1 = features1.get('glca_features')
        glca_feat2 = features2.get('glca_features')
        if glca_feat1 is not None and glca_feat2 is not None:
            glca_similarity = F.cosine_similarity(glca_feat1, glca_feat2, dim=1).unsqueeze(-1)  # Add dimension back
            # Don't normalize to [0,1] for BCEWithLogitsLoss - use raw similarity as logits
            # Ensure compatible shapes for BCE loss
            glca_similarity_squeezed = glca_similarity.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            targets_for_glca = targets.squeeze(-1) if targets.dim() > 1 else targets  # Ensure 1D
            glca_loss = self.bce_loss(glca_similarity_squeezed, targets_for_glca)
            branch_losses['glca'] = (glca_loss, {'loss': glca_loss.item()})
        
        # PWCA branch loss (training only)
        pwca_feat1 = features1.get('pwca_features')
        pwca_feat2 = features2.get('pwca_features')
        if pwca_feat1 is not None and pwca_feat2 is not None:
            pwca_similarity = F.cosine_similarity(pwca_feat1, pwca_feat2, dim=1).unsqueeze(-1)  # Add dimension back
            # Don't normalize to [0,1] for BCEWithLogitsLoss - use raw similarity as logits
            # Ensure compatible shapes for BCE loss
            pwca_similarity_squeezed = pwca_similarity.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            targets_for_pwca = targets.squeeze(-1) if targets.dim() > 1 else targets  # Ensure 1D
            pwca_loss = self.bce_loss(pwca_similarity_squeezed, targets_for_pwca)
            branch_losses['pwca'] = (pwca_loss, {'loss': pwca_loss.item()})
        
        return branch_losses
    
    def _compute_regularization_loss(self, model_outputs: Dict[str, Any]) -> torch.Tensor:
        """Compute regularization losses"""
        reg_loss = torch.tensor(0.0)
        
        # Add attention diversity regularization
        features1 = model_outputs.get('features1', {})
        attention_rollout = features1.get('attention_rollout')
        
        if attention_rollout is not None:
            # Encourage attention diversity (entropy regularization)
            batch_size = attention_rollout.size(0)
            seq_len = attention_rollout.size(1)
            
            # Attention entropy (higher entropy = more diverse attention)
            cls_attention = attention_rollout[:, 0, 1:]  # CLS attention to patches
            attention_entropy = -(cls_attention * torch.log(cls_attention + 1e-8)).sum(dim=1)
            target_entropy = math.log(seq_len - 1)  # Maximum entropy
            
            # Penalty for low entropy (too focused attention)
            entropy_penalty = F.relu(target_entropy * 0.5 - attention_entropy).mean()
            reg_loss += entropy_penalty
        
        return reg_loss
    
    def _update_adaptive_weights(self, loss_stats: Dict[str, float], epoch: int):
        """Update adaptive loss weights based on loss magnitudes"""
        if epoch < 10:  # Warmup period with fixed weights
            return
        
        # Get current loss values
        current_losses = {
            'bce': loss_stats.get('bce_loss', 0.0),
            'focal': loss_stats.get('focal_focal_loss', 0.0),
            'triplet': loss_stats.get('triplet_triplet_loss', 0.0)
        }
        
        # Compute relative loss magnitudes
        total_magnitude = sum(current_losses.values()) + 1e-8
        
        for loss_name, loss_value in current_losses.items():
            if loss_name in self.adaptive_weights:
                # Higher loss magnitude -> lower weight (balance the losses)
                relative_magnitude = loss_value / total_magnitude
                target_weight = 1.0 / (relative_magnitude + 1e-8)
                
                # Exponential moving average
                self.adaptive_weights[loss_name] = (
                    self.weight_momentum * self.adaptive_weights[loss_name] +
                    (1 - self.weight_momentum) * target_weight
                )


class VerificationLoss(nn.Module):
    """
    Main verification loss class that wraps all loss functions
    Provides a unified interface for training
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.LOSS_TYPE == "combined":
            self.loss_fn = CombinedLoss(config)
        elif config.LOSS_TYPE == "triplet":
            self.loss_fn = TripletLoss(margin=config.TRIPLET_MARGIN)
        elif config.LOSS_TYPE == "focal":
            self.loss_fn = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        elif config.LOSS_TYPE == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for autocast safety
        else:
            raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")
    
    def forward(self, model_outputs: Dict[str, Any], targets: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute verification loss
        
        Args:
            model_outputs: Model outputs dictionary
            targets: Ground truth labels
            **kwargs: Additional arguments (e.g., epoch for adaptive weighting)
            
        Returns:
            loss: Computed loss
            stats: Loss statistics dictionary
        """
        if isinstance(self.loss_fn, CombinedLoss):
            return self.loss_fn(model_outputs, targets, **kwargs)
        elif isinstance(self.loss_fn, (TripletLoss, FocalLoss)):
            # Extract verification scores for simple losses
            verification_score = model_outputs.get('verification_score')
            if verification_score is not None:
                return self.loss_fn(verification_score, targets)
            else:
                return torch.tensor(0.0), {}
        else:
            # Simple BCE loss
            verification_score = model_outputs.get('verification_score')
            if verification_score is not None:
                loss = self.loss_fn(verification_score, targets)
                return loss, {'bce_loss': loss.item()}
            else:
                return torch.tensor(0.0), {} 