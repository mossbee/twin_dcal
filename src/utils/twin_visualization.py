"""
Twin Face Verification Visualization Utilities

This module provides visualization tools for:
1. Attention maps (Self-attention, GLCA, PWCA)
2. Verification results and confusion analysis
3. Feature embeddings and similarity distributions
4. Training progress and loss curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AttentionVisualizer:
    """
    Visualizer for attention maps and model interpretability
    """
    
    def __init__(self, 
                 patch_size: int = 16,
                 input_size: int = 448,
                 save_dir: str = "visualizations"):
        self.patch_size = patch_size
        self.input_size = input_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate grid dimensions
        self.grid_size = input_size // patch_size  # 28 for 448//16
        
        # Color maps for different attention types
        self.attention_cmaps = {
            'self_attention': 'Blues',
            'glca': 'Reds', 
            'attention_rollout': 'Greens'
        }
    
    def visualize_attention_maps(self,
                                image: Union[torch.Tensor, np.ndarray, Image.Image],
                                attention_weights: Dict[str, torch.Tensor],
                                save_path: Optional[str] = None,
                                show_top_patches: int = 10) -> plt.Figure:
        """
        Visualize attention maps overlaid on input image
        
        Args:
            image: Input image [C, H, W] or PIL Image
            attention_weights: Dictionary with attention maps
            save_path: Path to save visualization
            show_top_patches: Number of top attention patches to highlight
            
        Returns:
            Figure object
        """
        # Prepare image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)  # Remove batch dimension
            # Convert to PIL for easier manipulation
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = (image * 255).clamp(0, 255).byte().cpu().numpy()
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Resize image to input size
        image = image.resize((self.input_size, self.input_size))
        
        # Create subplots
        num_attention_types = len(attention_weights)
        fig, axes = plt.subplots(2, num_attention_types, figsize=(5*num_attention_types, 10))
        if num_attention_types == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (attention_name, attention_map) in enumerate(attention_weights.items()):
            # Process attention map
            processed_attention = self._process_attention_map(attention_map)
            
            # Original image
            axes[0, idx].imshow(image)
            axes[0, idx].set_title(f'Original Image')
            axes[0, idx].axis('off')
            
            # Attention overlay
            attention_overlay = self._create_attention_overlay(
                image, processed_attention, 
                cmap=self.attention_cmaps.get(attention_name, 'viridis')
            )
            
            axes[1, idx].imshow(attention_overlay)
            axes[1, idx].set_title(f'{attention_name.replace("_", " ").title()} Attention')
            axes[1, idx].axis('off')
            
            # Highlight top patches
            if show_top_patches > 0:
                self._highlight_top_patches(
                    axes[1, idx], processed_attention, show_top_patches
                )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_verification_comparison(self,
                                        img1: Union[torch.Tensor, Image.Image],
                                        img2: Union[torch.Tensor, Image.Image],
                                        prediction: float,
                                        ground_truth: int,
                                        attention_weights1: Dict[str, torch.Tensor],
                                        attention_weights2: Dict[str, torch.Tensor],
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize verification comparison between two images
        """
        # Prepare images
        img1 = self._prepare_image_for_vis(img1)
        img2 = self._prepare_image_for_vis(img2)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Main images
        plt.subplot(3, 4, 1)
        plt.imshow(img1)
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(img2)
        plt.title('Image 2')
        plt.axis('off')
        
        # Verification result
        plt.subplot(3, 4, 3)
        plt.text(0.5, 0.7, f'Prediction: {prediction:.3f}', 
                ha='center', va='center', fontsize=14, weight='bold')
        plt.text(0.5, 0.5, f'Ground Truth: {"Same" if ground_truth else "Different"}',
                ha='center', va='center', fontsize=14)
        
        # Determine correctness
        is_correct = (prediction >= 0.5) == bool(ground_truth)
        result_color = 'green' if is_correct else 'red'
        result_text = 'CORRECT' if is_correct else 'INCORRECT'
        
        plt.text(0.5, 0.3, result_text, ha='center', va='center', 
                fontsize=16, weight='bold', color=result_color)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # Similarity bar
        plt.subplot(3, 4, 4)
        self._plot_similarity_bar(prediction, ground_truth)
        
        # Attention maps for both images
        attention_types = ['self_attention', 'glca', 'attention_rollout']
        for i, att_type in enumerate(attention_types):
            if att_type in attention_weights1:
                # Image 1 attention
                plt.subplot(3, 4, 5 + i)
                att1 = self._process_attention_map(attention_weights1[att_type])
                overlay1 = self._create_attention_overlay(img1, att1)
                plt.imshow(overlay1)
                plt.title(f'Img1: {att_type.replace("_", " ").title()}')
                plt.axis('off')
                
                # Image 2 attention
                plt.subplot(3, 4, 9 + i)
                att2 = self._process_attention_map(attention_weights2[att_type])
                overlay2 = self._create_attention_overlay(img2, att2)
                plt.imshow(overlay2)
                plt.title(f'Img2: {att_type.replace("_", " ").title()}')
                plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_evolution(self,
                                    image: Union[torch.Tensor, Image.Image],
                                    layer_attentions: List[torch.Tensor],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize how attention evolves across transformer layers
        """
        image = self._prepare_image_for_vis(image)
        num_layers = len(layer_attentions)
        
        # Select key layers to show
        layer_indices = np.linspace(0, num_layers-1, min(6, num_layers), dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, layer_idx in enumerate(layer_indices):
            attention_map = self._process_attention_map(layer_attentions[layer_idx])
            overlay = self._create_attention_overlay(image, attention_map)
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Layer {layer_idx + 1}')
            axes[i].axis('off')
        
        plt.suptitle('Attention Evolution Across Layers')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _prepare_image_for_vis(self, image: Union[torch.Tensor, Image.Image]) -> Image.Image:
        """Convert image to PIL Image for visualization"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            
            # Denormalize if needed (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            image = (image * 255).byte().cpu().numpy()
            image = Image.fromarray(image)
        
        return image.resize((self.input_size, self.input_size))
    
    def _process_attention_map(self, attention: torch.Tensor) -> np.ndarray:
        """Process attention tensor to 2D map"""
        if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
            attention = attention.mean(dim=1).squeeze(0)  # Average heads
        elif attention.dim() == 3:
            if attention.shape[0] == 1:  # Batch dimension
                attention = attention.squeeze(0)
        
        # Extract CLS attention to patches
        if attention.shape[0] == attention.shape[1]:  # Square attention matrix
            cls_attention = attention[0, 1:]  # CLS to patches (skip CLS token)
        else:
            cls_attention = attention[0]  # Already extracted
        
        # Reshape to 2D grid
        grid_attention = cls_attention.reshape(self.grid_size, self.grid_size)
        
        # Convert to numpy and normalize
        grid_attention = grid_attention.detach().cpu().numpy()
        grid_attention = (grid_attention - grid_attention.min()) / (grid_attention.max() - grid_attention.min() + 1e-8)
        
        return grid_attention
    
    def _create_attention_overlay(self, 
                                 image: Image.Image, 
                                 attention_map: np.ndarray,
                                 alpha: float = 0.6,
                                 cmap: str = 'jet') -> np.ndarray:
        """Create attention overlay on image"""
        # Resize attention map to image size
        attention_resized = cv2.resize(attention_map, (self.input_size, self.input_size))
        
        # Create heatmap
        cmap_obj = plt.cm.get_cmap(cmap)
        heatmap = cmap_obj(attention_resized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert image to numpy
        image_np = np.array(image)
        
        # Blend image and heatmap
        overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    def _highlight_top_patches(self, ax, attention_map: np.ndarray, top_k: int):
        """Highlight top-k attention patches"""
        # Find top-k patches
        flat_attention = attention_map.flatten()
        top_indices = np.argpartition(flat_attention, -top_k)[-top_k:]
        
        # Convert to 2D coordinates
        top_coords = np.unravel_index(top_indices, attention_map.shape)
        
        # Scale to image coordinates
        patch_size_vis = self.input_size / self.grid_size
        
        for y, x in zip(top_coords[0], top_coords[1]):
            rect = plt.Rectangle(
                (x * patch_size_vis, y * patch_size_vis),
                patch_size_vis, patch_size_vis,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
    
    def _plot_similarity_bar(self, prediction: float, ground_truth: int):
        """Plot similarity as a horizontal bar"""
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        # Create bar
        y_pos = 0.5
        bar_height = 0.3
        
        # Background bar
        plt.barh(y_pos, 1.0, height=bar_height, color='lightgray', alpha=0.3)
        
        # Prediction bar
        color_idx = min(int(prediction * len(colors)), len(colors) - 1)
        plt.barh(y_pos, prediction, height=bar_height, color=colors[color_idx])
        
        # Threshold line
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
        
        # Ground truth marker
        gt_x = 1.0 if ground_truth else 0.0
        plt.scatter(gt_x, y_pos, s=100, c='blue', marker='*', zorder=5)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Similarity Score')
        plt.title('Prediction vs Ground Truth')
        plt.grid(True, alpha=0.3)


def plot_attention_maps(image: torch.Tensor,
                       attention_weights: Dict[str, torch.Tensor],
                       save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function for plotting attention maps"""
    visualizer = AttentionVisualizer()
    return visualizer.visualize_attention_maps(image, attention_weights, save_path)


def visualize_verification_results(predictions: np.ndarray,
                                  labels: np.ndarray,
                                  save_dir: str = "verification_analysis") -> Dict[str, plt.Figure]:
    """
    Create comprehensive visualization of verification results
    
    Returns:
        Dictionary of figure names and objects
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # 1. Score distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    positive_scores = predictions[labels == 1]
    negative_scores = predictions[labels == 0]
    
    ax1.hist(negative_scores, bins=50, alpha=0.7, label='Different Person', 
            color='red', density=True)
    ax1.hist(positive_scores, bins=50, alpha=0.7, label='Same Person', 
            color='green', density=True)
    ax1.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Similarity Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    figures['score_distribution'] = fig1
    
    # 2. Confusion matrix
    from sklearn.metrics import confusion_matrix
    binary_pred = (predictions >= 0.5).astype(int)
    cm = confusion_matrix(labels, binary_pred)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    ax2.set_xticklabels(['Different', 'Same'])
    ax2.set_yticklabels(['Different', 'Same'])
    
    figures['confusion_matrix'] = fig2
    
    # 3. ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    
    figures['roc_curve'] = fig3
    
    # 4. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend(loc="lower left")
    ax4.grid(True, alpha=0.3)
    
    figures['pr_curve'] = fig4
    
    # Save all figures
    for name, fig in figures.items():
        save_path = save_dir / f"{name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return figures


def visualize_feature_space(features: np.ndarray,
                           labels: np.ndarray,
                           method: str = 'tsne',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize feature embeddings in 2D space using t-SNE or PCA
    
    Args:
        features: Feature vectors [N, D]
        labels: Binary labels [N]
        method: 'tsne' or 'pca'
        save_path: Path to save visualization
        
    Returns:
        Figure object
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    # Reduce dimensions
    features_2d = reducer.fit_transform(features)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot different classes
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = 'Same Person' if label == 1 else 'Different Person'
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=colors[i % len(colors)], label=label_name, alpha=0.6)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Feature Space Visualization ({method.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_training_dashboard(metrics_history: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive training dashboard
    
    Args:
        metrics_history: Dictionary with metric names and their values over epochs
        save_path: Path to save dashboard
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define which metrics to plot
    plot_configs = [
        ('total_loss', 'Total Loss', 'Loss'),
        ('verification_accuracy', 'Verification Accuracy', 'Accuracy'),
        ('roc_auc', 'ROC AUC', 'AUC'),
        ('eer', 'Equal Error Rate', 'EER'),
        ('learning_rate', 'Learning Rate', 'Learning Rate'),
        ('grad_norm', 'Gradient Norm', 'Norm')
    ]
    
    for i, (metric_name, title, ylabel) in enumerate(plot_configs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot train and validation if available
        train_key = f'train_{metric_name}'
        val_key = f'val_{metric_name}'
        
        if train_key in metrics_history:
            epochs = range(len(metrics_history[train_key]))
            ax.plot(epochs, metrics_history[train_key], label='Train', color='blue')
        
        if val_key in metrics_history:
            epochs = range(len(metrics_history[val_key]))
            ax.plot(epochs, metrics_history[val_key], label='Validation', color='red')
        
        # Fallback to just the metric name
        elif metric_name in metrics_history:
            epochs = range(len(metrics_history[metric_name]))
            ax.plot(epochs, metrics_history[metric_name], color='blue')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_verification_examples(model_outputs: Dict[str, Any],
                              dataset: Any,
                              save_dir: str = "verification_examples",
                              num_examples: int = 20):
    """
    Save examples of verification results (correct and incorrect)
    
    Args:
        model_outputs: Dictionary with predictions, paths, etc.
        dataset: Dataset object for loading images
        save_dir: Directory to save examples
        num_examples: Number of examples to save per category
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = model_outputs['predictions']
    labels = model_outputs['labels']
    paths = model_outputs.get('paths', [])
    
    if not paths:
        print("Warning: No image paths provided, cannot save examples")
        return
    
    # Categorize results
    binary_pred = (predictions >= 0.5).astype(int)
    correct_mask = (binary_pred == labels)
    
    # True positives (correct same person)
    tp_mask = correct_mask & (labels == 1)
    # True negatives (correct different person)
    tn_mask = correct_mask & (labels == 0)
    # False positives (incorrectly predicted same)
    fp_mask = (~correct_mask) & (labels == 0)
    # False negatives (incorrectly predicted different)
    fn_mask = (~correct_mask) & (labels == 1)
    
    categories = {
        'true_positives': tp_mask,
        'true_negatives': tn_mask,
        'false_positives': fp_mask,
        'false_negatives': fn_mask
    }
    
    visualizer = AttentionVisualizer()
    
    for category, mask in categories.items():
        if not np.any(mask):
            continue
            
        category_dir = save_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Get indices for this category
        indices = np.where(mask)[0]
        
        # Sample examples
        num_samples = min(num_examples, len(indices))
        sampled_indices = np.random.choice(indices, num_samples, replace=False)
        
        for i, idx in enumerate(sampled_indices):
            try:
                # Load images
                path1, path2 = paths[idx]
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
                
                # Create comparison visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img1)
                axes[0].set_title('Image 1')
                axes[0].axis('off')
                
                axes[1].imshow(img2)
                axes[1].set_title('Image 2')
                axes[1].axis('off')
                
                # Result info
                pred_score = predictions[idx]
                gt_label = labels[idx]
                
                axes[2].text(0.5, 0.7, f'Prediction: {pred_score:.3f}', 
                           ha='center', va='center', fontsize=12)
                axes[2].text(0.5, 0.5, f'Ground Truth: {"Same" if gt_label else "Different"}',
                           ha='center', va='center', fontsize=12)
                axes[2].text(0.5, 0.3, category.replace('_', ' ').title(),
                           ha='center', va='center', fontsize=14, weight='bold')
                axes[2].set_xlim(0, 1)
                axes[2].set_ylim(0, 1)
                axes[2].axis('off')
                
                # Save
                save_path = category_dir / f'example_{i:03d}_score_{pred_score:.3f}.png'
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error processing example {i} in {category}: {e}")
    
    print(f"Verification examples saved to {save_dir}") 