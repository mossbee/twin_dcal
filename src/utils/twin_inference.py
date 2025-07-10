"""
Twin Face Verification Inference Utilities

This module provides utilities for:
1. Loading trained models for inference
2. Batch verification of face pairs
3. Feature extraction and similarity computation
4. Threshold optimization and calibration
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.dcal_verification_model import create_dcal_model, DCALVerificationModel


class TwinInferenceEngine:
    """
    Inference engine for twin face verification
    Handles model loading, preprocessing, and batch inference
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device to run inference on
            threshold: Verification threshold
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model and config
        self.model = None
        self.config = None
        self.transform = None
        
        self._load_model()
        self._setup_transforms()
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Verification threshold: {self.threshold}")
    
    def _load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load config
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Try to get config from checkpoint
            config_dict = checkpoint.get('config', {})
        
        # Create config object (simplified version)
        self.config = SimpleConfig(config_dict)
        
        # Create model
        self.model = create_dcal_model(self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP wrapped models
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"  Best metric: {checkpoint['best_metric']:.4f}")
    
    def _setup_transforms(self):
        """Setup preprocessing transforms"""
        input_size = getattr(self.config, 'INPUT_SIZE', 448)
        mean = getattr(self.config, 'MEAN', [0.485, 0.456, 0.406])
        std = getattr(self.config, 'STD', [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def verify_pair(self, 
                   img1: Union[str, Image.Image, torch.Tensor],
                   img2: Union[str, Image.Image, torch.Tensor],
                   return_features: bool = False,
                   return_attention: bool = False) -> Dict[str, Any]:
        """
        Verify if two images belong to the same person
        
        Args:
            img1: First image (path, PIL Image, or tensor)
            img2: Second image (path, PIL Image, or tensor)
            return_features: Whether to return feature embeddings
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with verification results
        """
        # Preprocess images
        tensor1 = self._preprocess_image(img1)
        tensor2 = self._preprocess_image(img2)
        
        # Add batch dimension
        tensor1 = tensor1.unsqueeze(0).to(self.device)
        tensor2 = tensor2.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                tensor1, tensor2, 
                training=False,
                return_features=return_features or return_attention
            )
        
        # Extract results
        verification_score = outputs['verification_score'].cpu().item()
        prediction = verification_score >= self.threshold
        
        result = {
            'verification_score': verification_score,
            'prediction': prediction,
            'same_person': prediction
        }
        
        # Add features if requested
        if return_features:
            features1 = outputs['features1']
            features2 = outputs['features2']
            
            result.update({
                'features1': {
                    key: tensor.cpu().numpy() for key, tensor in features1.items()
                    if isinstance(tensor, torch.Tensor)
                },
                'features2': {
                    key: tensor.cpu().numpy() for key, tensor in features2.items()
                    if isinstance(tensor, torch.Tensor)
                }
            })
        
        # Add attention maps if requested
        if return_attention:
            features1 = outputs.get('features1', {})
            features2 = outputs.get('features2', {})
            
            attention_data = {}
            
            # Attention rollout
            if 'attention_rollout' in features1:
                attention_data['attention_rollout1'] = features1['attention_rollout'].cpu()
            if 'attention_rollout' in features2:
                attention_data['attention_rollout2'] = features2['attention_rollout'].cpu()
            
            # GLCA attention
            if 'glca_attention' in features1:
                attention_data['glca_attention1'] = features1['glca_attention'].cpu()
            if 'glca_attention' in features2:
                attention_data['glca_attention2'] = features2['glca_attention'].cpu()
            
            result['attention_maps'] = attention_data
        
        return result
    
    def batch_verify(self, 
                    image_pairs: List[Tuple[str, str]],
                    batch_size: int = 16,
                    show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Verify multiple image pairs in batches
        
        Args:
            image_pairs: List of (image1_path, image2_path) tuples
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of verification results
        """
        results = []
        
        # Process in batches
        num_batches = (len(image_pairs) + batch_size - 1) // batch_size
        iterator = range(num_batches)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Verifying pairs")
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_pairs))
            batch_pairs = image_pairs[start_idx:end_idx]
            
            # Process batch
            batch_results = self._process_batch(batch_pairs)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, image_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of image pairs"""
        # Load and preprocess images
        img1_batch = []
        img2_batch = []
        
        for img1_path, img2_path in image_pairs:
            try:
                tensor1 = self._preprocess_image(img1_path)
                tensor2 = self._preprocess_image(img2_path)
                img1_batch.append(tensor1)
                img2_batch.append(tensor2)
            except Exception as e:
                print(f"Error loading images {img1_path}, {img2_path}: {e}")
                # Add dummy tensors
                dummy_tensor = torch.zeros(3, 448, 448)
                img1_batch.append(dummy_tensor)
                img2_batch.append(dummy_tensor)
        
        # Stack into batch tensors
        img1_batch = torch.stack(img1_batch).to(self.device)
        img2_batch = torch.stack(img2_batch).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img1_batch, img2_batch, training=False)
        
        # Process results
        verification_scores = outputs['verification_score'].cpu().numpy()
        predictions = verification_scores >= self.threshold
        
        results = []
        for i, (img1_path, img2_path) in enumerate(image_pairs):
            results.append({
                'img1_path': img1_path,
                'img2_path': img2_path,
                'verification_score': float(verification_scores[i]),
                'prediction': bool(predictions[i]),
                'same_person': bool(predictions[i])
            })
        
        return results
    
    def extract_features(self, 
                        images: List[Union[str, Image.Image]],
                        batch_size: int = 32,
                        feature_type: str = 'combined') -> np.ndarray:
        """
        Extract features from images
        
        Args:
            images: List of images (paths or PIL Images)
            batch_size: Batch size for processing
            feature_type: 'sa', 'glca', or 'combined'
            
        Returns:
            Feature matrix [N, D]
        """
        features_list = []
        
        # Process in batches
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                try:
                    tensor = self._preprocess_image(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error loading image {img}: {e}")
                    dummy_tensor = torch.zeros(3, 448, 448)
                    batch_tensors.append(dummy_tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                # Use dummy second image for feature extraction
                dummy_img = torch.zeros_like(batch_tensor)
                outputs = self.model(batch_tensor, dummy_img, training=False, return_features=True)
                
                features1 = outputs['features1']
                
                if feature_type == 'sa':
                    batch_features = features1['sa_features'].cpu().numpy()
                elif feature_type == 'glca':
                    batch_features = features1['glca_features'].cpu().numpy()
                else:  # combined
                    batch_features = features1['combined_features'].cpu().numpy()
                
                features_list.append(batch_features)
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=0)
        return all_features
    
    def compute_similarity_matrix(self, 
                                 images: List[Union[str, Image.Image]],
                                 batch_size: int = 32) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a set of images
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            Similarity matrix [N, N]
        """
        # Extract features
        features = self.extract_features(images, batch_size)
        
        # Compute pairwise similarities
        # Normalize features for cosine similarity
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(features_norm, features_norm.T)
        
        return similarity_matrix
    
    def optimize_threshold(self, 
                          validation_pairs: List[Tuple[str, str, int]],
                          metric: str = 'f1') -> float:
        """
        Optimize verification threshold on validation data
        
        Args:
            validation_pairs: List of (img1_path, img2_path, label) tuples
            metric: Metric to optimize ('f1', 'accuracy', 'eer')
            
        Returns:
            Optimal threshold
        """
        print(f"Optimizing threshold on {len(validation_pairs)} validation pairs...")
        
        # Get predictions for all pairs
        image_pairs = [(pair[0], pair[1]) for pair in validation_pairs]
        labels = np.array([pair[2] for pair in validation_pairs])
        
        results = self.batch_verify(image_pairs, show_progress=True)
        predictions = np.array([r['verification_score'] for r in results])
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            binary_pred = (predictions >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(labels, binary_pred)
            elif metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(labels, binary_pred)
            elif metric == 'eer':
                # For EER, we want to minimize it
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(labels, predictions)
                fnr = 1 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                score = 1 - (fpr[eer_idx] + fnr[eer_idx]) / 2  # Convert to maximization
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.3f} ({metric}: {best_score:.4f})")
        self.threshold = best_threshold
        
        return best_threshold
    
    def _preprocess_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Preprocess image for inference"""
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            # Apply transforms
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            # Assume already preprocessed
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def set_threshold(self, threshold: float):
        """Set verification threshold"""
        self.threshold = threshold
        print(f"Verification threshold updated to {threshold:.3f}")


class SimpleConfig:
    """Simple configuration class for inference"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Set defaults
        self.INPUT_SIZE = 448
        self.D_MODEL = 768
        self.NUM_HEADS = 12
        self.SA_BLOCKS = 12
        self.GLCA_BLOCKS = 1
        self.LOCAL_QUERY_RATIO = 0.15
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        
        # Update with provided config
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_model_for_inference(model_path: str, 
                           config_path: Optional[str] = None,
                           device: Optional[str] = None) -> TwinInferenceEngine:
    """
    Convenience function to load model for inference
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        device: Device to load model on
        
    Returns:
        Configured inference engine
    """
    return TwinInferenceEngine(model_path, config_path, device)


def batch_verification(image_pairs: List[Tuple[str, str]],
                      model_path: str,
                      batch_size: int = 16,
                      threshold: float = 0.5,
                      output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for batch verification
    
    Args:
        image_pairs: List of (img1_path, img2_path) tuples
        model_path: Path to trained model
        batch_size: Batch size for processing
        threshold: Verification threshold
        output_path: Optional path to save results
        
    Returns:
        List of verification results
    """
    # Load model
    engine = TwinInferenceEngine(model_path, threshold=threshold)
    
    # Process pairs
    results = engine.batch_verify(image_pairs, batch_size)
    
    # Save results if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def evaluate_on_test_set(test_pairs: List[Tuple[str, str, int]],
                        model_path: str,
                        batch_size: int = 16) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        test_pairs: List of (img1_path, img2_path, label) tuples
        model_path: Path to trained model
        batch_size: Batch size for processing
        
    Returns:
        Evaluation metrics
    """
    from .verification_metrics import VerificationMetrics
    
    # Load model
    engine = TwinInferenceEngine(model_path)
    
    # Get predictions
    image_pairs = [(pair[0], pair[1]) for pair in test_pairs]
    labels = np.array([pair[2] for pair in test_pairs])
    paths = [(pair[0], pair[1]) for pair in test_pairs]
    
    results = engine.batch_verify(image_pairs, batch_size, show_progress=True)
    predictions = np.array([r['verification_score'] for r in results])
    
    # Compute metrics
    metrics_calculator = VerificationMetrics()
    metrics = metrics_calculator.compute_all_metrics(predictions, labels, paths)
    
    return metrics


def extract_and_save_features(images: List[str],
                             model_path: str,
                             output_path: str,
                             batch_size: int = 32,
                             feature_type: str = 'combined'):
    """
    Extract features from images and save to file
    
    Args:
        images: List of image paths
        model_path: Path to trained model
        output_path: Path to save features
        batch_size: Batch size for processing
        feature_type: Type of features to extract
    """
    # Load model
    engine = TwinInferenceEngine(model_path)
    
    # Extract features
    features = engine.extract_features(images, batch_size, feature_type)
    
    # Save features
    np.savez(output_path, 
             features=features,
             image_paths=images,
             feature_type=feature_type)
    
    print(f"Features saved to {output_path}")
    print(f"Feature shape: {features.shape}")


def benchmark_inference_speed(model_path: str,
                             image_size: Tuple[int, int] = (448, 448),
                             batch_sizes: List[int] = [1, 4, 8, 16, 32],
                             num_iterations: int = 100) -> Dict[int, float]:
    """
    Benchmark inference speed for different batch sizes
    
    Args:
        model_path: Path to trained model
        image_size: Input image size
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations per batch size
        
    Returns:
        Dictionary mapping batch size to average inference time
    """
    # Load model
    engine = TwinInferenceEngine(model_path)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Create dummy data
        dummy_img1 = torch.randn(batch_size, 3, *image_size).to(engine.device)
        dummy_img2 = torch.randn(batch_size, 3, *image_size).to(engine.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = engine.model(dummy_img1, dummy_img2, training=False)
        
        # Benchmark
        torch.cuda.synchronize() if engine.device.startswith('cuda') else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = engine.model(dummy_img1, dummy_img2, training=False)
        
        torch.cuda.synchronize() if engine.device.startswith('cuda') else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        results[batch_size] = avg_time
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Throughput: {batch_size / avg_time:.2f} pairs/s")
    
    return results 