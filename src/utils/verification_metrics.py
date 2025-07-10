"""
Verification Metrics for Twin Face Verification

This module implements comprehensive evaluation metrics:
1. ROC curves and AUC
2. Equal Error Rate (EER)
3. True Accept Rate at False Accept Rate
4. Precision-Recall curves
5. Twin-specific accuracy metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    accuracy_score, f1_score, confusion_matrix,
    average_precision_score
)
import torch


class VerificationMetrics:
    """
    Comprehensive metrics computation for face verification
    """
    
    def __init__(self, twin_pairs_info: Optional[List[List[str]]] = None):
        self.twin_pairs_info = twin_pairs_info or []
        self.twin_pairs_set = set()
        
        # Convert twin pairs to set for fast lookup
        for pair in self.twin_pairs_info:
            self.twin_pairs_set.add((pair[0], pair[1]))
            self.twin_pairs_set.add((pair[1], pair[0]))
    
    def compute_all_metrics(self, 
                           predictions: np.ndarray,
                           labels: np.ndarray,
                           paths: Optional[List[Tuple[str, str]]] = None) -> Dict[str, float]:
        """
        Compute all verification metrics
        
        Args:
            predictions: Predicted similarity scores [N]
            labels: Ground truth labels (1=same, 0=different) [N]
            paths: Optional image paths for twin-specific metrics
            
        Returns:
            Dictionary with all computed metrics
        """
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()
        
        metrics = {}
        
        # ROC metrics
        roc_metrics = self.compute_roc_metrics(predictions, labels)
        metrics.update(roc_metrics)
        
        # Precision-Recall metrics
        pr_metrics = self.compute_precision_recall_metrics(predictions, labels)
        metrics.update(pr_metrics)
        
        # Threshold-based metrics
        threshold_metrics = self.compute_threshold_metrics(predictions, labels)
        metrics.update(threshold_metrics)
        
        # Twin-specific metrics (if paths provided)
        if paths and self.twin_pairs_info:
            twin_metrics = self.compute_twin_metrics(predictions, labels, paths)
            metrics.update(twin_metrics)
        
        return metrics
    
    def compute_roc_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute ROC-based metrics"""
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # True Accept Rate at specific False Accept Rates
        tar_at_far_001 = self._compute_tar_at_far(fpr, tpr, 0.001)
        tar_at_far_01 = self._compute_tar_at_far(fpr, tpr, 0.01)
        tar_at_far_1 = self._compute_tar_at_far(fpr, tpr, 0.1)
        
        return {
            'roc_auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'tar_at_far_0.001': tar_at_far_001,
            'tar_at_far_0.01': tar_at_far_01,
            'tar_at_far_0.1': tar_at_far_1
        }
    
    def compute_precision_recall_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute Precision-Recall based metrics"""
        precision, recall, pr_thresholds = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        average_precision = average_precision_score(labels, predictions)
        
        # Find best F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_f1_threshold = pr_thresholds[best_f1_idx] if len(pr_thresholds) > best_f1_idx else 0.5
        
        return {
            'pr_auc': pr_auc,
            'average_precision': average_precision,
            'best_f1': best_f1,
            'best_f1_threshold': best_f1_threshold
        }
    
    def compute_threshold_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute metrics for different thresholds"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_metrics = {}
        
        for threshold in thresholds:
            binary_pred = (predictions >= threshold).astype(int)
            
            accuracy = accuracy_score(labels, binary_pred)
            f1 = f1_score(labels, binary_pred)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, binary_pred).ravel()
            
            # Compute rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            
            threshold_metrics.update({
                f'accuracy_at_{threshold}': accuracy,
                f'f1_at_{threshold}': f1,
                f'precision_at_{threshold}': precision,
                f'recall_at_{threshold}': recall,
                f'tpr_at_{threshold}': tpr,
                f'fpr_at_{threshold}': fpr
            })
        
        # Find optimal threshold based on different criteria
        optimal_threshold_f1 = self._find_optimal_threshold(predictions, labels, 'f1')
        optimal_threshold_acc = self._find_optimal_threshold(predictions, labels, 'accuracy')
        
        threshold_metrics.update({
            'optimal_threshold_f1': optimal_threshold_f1,
            'optimal_threshold_accuracy': optimal_threshold_acc
        })
        
        return threshold_metrics
    
    def compute_twin_metrics(self, 
                            predictions: np.ndarray, 
                            labels: np.ndarray,
                            paths: List[Tuple[str, str]]) -> Dict[str, float]:
        """Compute metrics specifically for twin pairs"""
        twin_indices = []
        regular_indices = []
        
        for idx, (path1, path2) in enumerate(paths):
            # Extract identity IDs from paths
            id1 = self._extract_identity_from_path(path1)
            id2 = self._extract_identity_from_path(path2)
            
            if (id1, id2) in self.twin_pairs_set:
                twin_indices.append(idx)
            else:
                regular_indices.append(idx)
        
        twin_metrics = {}
        
        if len(twin_indices) > 0:
            twin_predictions = predictions[twin_indices]
            twin_labels = labels[twin_indices]
            
            # Twin-specific accuracy (harder cases)
            optimal_threshold = self._find_optimal_threshold(predictions, labels, 'f1')
            twin_binary_pred = (twin_predictions >= optimal_threshold).astype(int)
            twin_accuracy = accuracy_score(twin_labels, twin_binary_pred)
            twin_f1 = f1_score(twin_labels, twin_binary_pred)
            
            # Twin ROC AUC
            if len(np.unique(twin_labels)) > 1:  # Need both classes
                twin_roc_auc = auc(twin_labels, twin_predictions) # Corrected from roc_auc_score
            else:
                twin_roc_auc = 0.0
            
            twin_metrics.update({
                'twin_accuracy': twin_accuracy,
                'twin_f1': twin_f1,
                'twin_roc_auc': twin_roc_auc,
                'twin_samples': len(twin_indices)
            })
        
        if len(regular_indices) > 0:
            regular_predictions = predictions[regular_indices]
            regular_labels = labels[regular_indices]
            
            # Regular pairs accuracy
            optimal_threshold = self._find_optimal_threshold(predictions, labels, 'f1')
            regular_binary_pred = (regular_predictions >= optimal_threshold).astype(int)
            regular_accuracy = accuracy_score(regular_labels, regular_binary_pred)
            
            twin_metrics.update({
                'regular_accuracy': regular_accuracy,
                'regular_samples': len(regular_indices)
            })
        
        return twin_metrics
    
    def _compute_tar_at_far(self, fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
        """Compute True Accept Rate at given False Accept Rate"""
        # Find closest FAR to target
        far_idx = np.argmin(np.abs(fpr - target_far))
        tar = tpr[far_idx]
        return tar
    
    def _find_optimal_threshold(self, 
                               predictions: np.ndarray, 
                               labels: np.ndarray,
                               metric: str = 'f1') -> float:
        """Find optimal threshold based on specified metric"""
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            binary_pred = (predictions >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(labels, binary_pred)
            elif metric == 'accuracy':
                score = accuracy_score(labels, binary_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _extract_identity_from_path(self, path: str) -> str:
        """Extract identity ID from image path"""
        # This is dataset-specific - implement based on your path structure
        # Example: "/data/id_123/image.jpg" -> "id_123"
        parts = path.split('/')
        for part in parts:
            if part.startswith('id_'):
                return part
        return "unknown"
    
    def plot_roc_curve(self, 
                      predictions: np.ndarray, 
                      labels: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, 
                                   predictions: np.ndarray, 
                                   labels: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_score_distribution(self, 
                               predictions: np.ndarray, 
                               labels: np.ndarray,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of similarity scores"""
        positive_scores = predictions[labels == 1]
        negative_scores = predictions[labels == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(negative_scores, bins=50, alpha=0.7, label='Different Person', 
                color='red', density=True)
        plt.hist(positive_scores, bins=50, alpha=0.7, label='Same Person', 
                color='green', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


# Convenience functions
def compute_roc_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute ROC-based metrics"""
    metrics = VerificationMetrics()
    return metrics.compute_roc_metrics(predictions, labels)


def compute_verification_accuracy(predictions: np.ndarray, 
                                 labels: np.ndarray,
                                 threshold: float = 0.5) -> float:
    """Compute verification accuracy at given threshold"""
    binary_pred = (predictions >= threshold).astype(int)
    return accuracy_score(labels, binary_pred)


def find_optimal_threshold(predictions: np.ndarray, 
                          labels: np.ndarray,
                          metric: str = 'f1') -> float:
    """Find optimal threshold for verification"""
    metrics = VerificationMetrics()
    return metrics._find_optimal_threshold(predictions, labels, metric)


def evaluate_model_predictions(model_outputs: Dict[str, Any],
                              ground_truth: Dict[str, Any],
                              twin_pairs_info: Optional[List] = None) -> Dict[str, float]:
    """
    Evaluate model predictions comprehensively
    
    Args:
        model_outputs: Dictionary with 'predictions' and optional 'paths'
        ground_truth: Dictionary with 'labels'
        twin_pairs_info: Twin pairs information for twin-specific metrics
        
    Returns:
        Complete evaluation metrics
    """
    metrics_calculator = VerificationMetrics(twin_pairs_info)
    
    return metrics_calculator.compute_all_metrics(
        predictions=model_outputs['predictions'],
        labels=ground_truth['labels'],
        paths=model_outputs.get('paths')
    ) 