#!/usr/bin/env python3
"""
Feature Extraction Script for Twin Face Verification using DCAL

This script extracts features from face images using a trained DCAL model.
Supports batch processing and different feature types (SA, GLCA, combined).

Usage:
    # Extract features from a directory of images
    python extract_features.py --model best_model.pth --images_dir data/faces/ --output features.npz

    # Extract specific feature type
    python extract_features.py --model best_model.pth --images_list images.txt --feature_type sa

    # Extract features and compute similarity matrix
    python extract_features.py --model best_model.pth --images_dir data/ --similarity_matrix
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from utils.twin_inference import TwinInferenceEngine, extract_and_save_features
from utils.twin_visualization import visualize_feature_space


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract Features using DCAL Model')
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run extraction on (cuda/cpu)'
    )
    
    # Input data
    parser.add_argument(
        '--images_dir',
        type=str,
        default=None,
        help='Directory containing images to extract features from'
    )
    
    parser.add_argument(
        '--images_list',
        type=str,
        default=None,
        help='Text file with list of image paths'
    )
    
    parser.add_argument(
        '--image_extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='Image file extensions to process'
    )
    
    # Feature extraction options
    parser.add_argument(
        '--feature_type',
        type=str,
        default='combined',
        choices=['sa', 'glca', 'combined'],
        help='Type of features to extract'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='features.npz',
        help='Output file for features (.npz format)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for additional files'
    )
    
    parser.add_argument(
        '--save_individual',
        action='store_true',
        help='Save individual feature files for each image'
    )
    
    # Analysis options
    parser.add_argument(
        '--similarity_matrix',
        action='store_true',
        help='Compute and save pairwise similarity matrix'
    )
    
    parser.add_argument(
        '--visualize_features',
        action='store_true',
        help='Create 2D visualization of feature space'
    )
    
    parser.add_argument(
        '--top_k_similar',
        type=int,
        default=0,
        help='Find top-k most similar images for each image'
    )
    
    parser.add_argument(
        '--cluster_features',
        action='store_true',
        help='Perform clustering analysis on features'
    )
    
    return parser.parse_args()


def collect_image_paths(images_dir: Optional[str] = None,
                       images_list: Optional[str] = None,
                       extensions: List[str] = None) -> List[str]:
    """
    Collect image paths from directory or list file
    
    Args:
        images_dir: Directory containing images
        images_list: Text file with image paths
        extensions: Allowed file extensions
        
    Returns:
        List of image paths
    """
    image_paths = []
    
    if images_dir:
        images_dir = Path(images_dir)
        
        for ext in extensions:
            # Search recursively for images
            pattern = f"**/*{ext}"
            paths = list(images_dir.glob(pattern))
            image_paths.extend([str(p) for p in paths])
            
            # Also search for uppercase extensions
            pattern = f"**/*{ext.upper()}"
            paths = list(images_dir.glob(pattern))
            image_paths.extend([str(p) for p in paths])
    
    elif images_list:
        with open(images_list, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    image_paths.append(path)
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    # Verify images can be loaded
    valid_paths = []
    print("Verifying image files...")
    
    for path in tqdm(image_paths, desc="Checking images"):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Skipping invalid image {path}: {e}")
    
    return valid_paths


def extract_features_batch(engine: TwinInferenceEngine,
                          image_paths: List[str],
                          feature_type: str = 'combined',
                          batch_size: int = 32) -> np.ndarray:
    """
    Extract features from list of images
    
    Args:
        engine: Inference engine
        image_paths: List of image paths
        feature_type: Type of features to extract
        batch_size: Batch size for processing
        
    Returns:
        Feature matrix [N, D]
    """
    print(f"Extracting {feature_type} features from {len(image_paths)} images...")
    
    features = engine.extract_features(
        image_paths, 
        batch_size=batch_size, 
        feature_type=feature_type
    )
    
    print(f"Extracted features shape: {features.shape}")
    return features


def save_features(features: np.ndarray,
                 image_paths: List[str],
                 output_path: str,
                 feature_type: str,
                 additional_info: Optional[Dict[str, Any]] = None):
    """
    Save features to file
    
    Args:
        features: Feature matrix
        image_paths: List of image paths
        output_path: Output file path
        feature_type: Type of features
        additional_info: Additional metadata
    """
    print(f"Saving features to {output_path}...")
    
    # Prepare data to save
    save_data = {
        'features': features,
        'image_paths': image_paths,
        'feature_type': feature_type,
        'feature_dim': features.shape[1],
        'num_images': len(image_paths),
        'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if additional_info:
        save_data.update(additional_info)
    
    # Save as npz file
    np.savez_compressed(output_path, **save_data)
    
    print(f"Features saved: {features.shape[0]} images, {features.shape[1]} dimensions")


def save_individual_features(features: np.ndarray,
                           image_paths: List[str],
                           output_dir: str,
                           feature_type: str):
    """
    Save individual feature files for each image
    
    Args:
        features: Feature matrix
        image_paths: List of image paths
        output_dir: Output directory
        feature_type: Type of features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving individual feature files to {output_dir}...")
    
    for i, (image_path, feature_vec) in enumerate(tqdm(zip(image_paths, features), desc="Saving")):
        # Create filename based on image path
        image_name = Path(image_path).stem
        feature_file = output_dir / f"{image_name}_{feature_type}.npy"
        
        # Save feature vector
        np.save(feature_file, feature_vec)
    
    print(f"Saved {len(image_paths)} individual feature files")


def compute_similarity_matrix(features: np.ndarray,
                            image_paths: List[str],
                            output_path: str) -> np.ndarray:
    """
    Compute and save pairwise similarity matrix
    
    Args:
        features: Feature matrix
        image_paths: Image paths
        output_path: Output file path
        
    Returns:
        Similarity matrix
    """
    print("Computing pairwise similarity matrix...")
    
    # Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    # Save similarity matrix
    print(f"Saving similarity matrix to {output_path}...")
    np.savez_compressed(
        output_path,
        similarity_matrix=similarity_matrix,
        image_paths=image_paths
    )
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix


def find_top_k_similar(similarity_matrix: np.ndarray,
                      image_paths: List[str],
                      k: int,
                      output_path: str):
    """
    Find top-k most similar images for each image
    
    Args:
        similarity_matrix: Precomputed similarity matrix
        image_paths: Image paths
        k: Number of similar images to find
        output_path: Output file path
    """
    print(f"Finding top-{k} similar images for each image...")
    
    # For each image, find top-k most similar (excluding itself)
    top_k_results = {}
    
    for i, query_path in enumerate(tqdm(image_paths, desc="Finding similar")):
        # Get similarities for this image (excluding itself)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        
        # Find top-k indices
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]
        
        # Store results
        top_k_results[query_path] = [
            {
                'path': image_paths[idx],
                'similarity': float(similarities[idx])
            }
            for idx in top_k_indices
        ]
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(top_k_results, f, indent=2)
    
    print(f"Top-{k} similar images saved to {output_path}")


def cluster_features(features: np.ndarray,
                    image_paths: List[str],
                    output_dir: str,
                    n_clusters: Optional[int] = None):
    """
    Perform clustering analysis on features
    
    Args:
        features: Feature matrix
        image_paths: Image paths
        output_dir: Output directory
        n_clusters: Number of clusters (auto-determine if None)
    """
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    
    print("Performing clustering analysis...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        print("Determining optimal number of clusters...")
        silhouette_scores = []
        cluster_range = range(2, min(20, len(features) // 2))
        
        for n in tqdm(cluster_range, desc="Testing clusters"):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            score = silhouette_score(features, cluster_labels)
            silhouette_scores.append(score)
        
        # Choose best number of clusters
        best_idx = np.argmax(silhouette_scores)
        n_clusters = list(cluster_range)[best_idx]
        print(f"Optimal number of clusters: {n_clusters}")
    
    # Perform K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Perform DBSCAN clustering for comparison
    print("Performing DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(features)
    
    # Organize results by cluster
    kmeans_clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in kmeans_clusters:
            kmeans_clusters[label] = []
        kmeans_clusters[label].append(image_paths[i])
    
    dbscan_clusters = {}
    for i, label in enumerate(dbscan_labels):
        if label not in dbscan_clusters:
            dbscan_clusters[label] = []
        dbscan_clusters[label].append(image_paths[i])
    
    # Save clustering results
    clustering_results = {
        'kmeans': {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'clusters': kmeans_clusters,
            'silhouette_score': float(silhouette_score(features, cluster_labels))
        },
        'dbscan': {
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'cluster_labels': dbscan_labels.tolist(),
            'clusters': dbscan_clusters,
            'silhouette_score': float(silhouette_score(features, dbscan_labels)) if len(set(dbscan_labels)) > 1 else 0.0
        }
    }
    
    # Save results
    results_file = output_dir / 'clustering_results.json'
    with open(results_file, 'w') as f:
        json.dump(clustering_results, f, indent=2)
    
    print(f"Clustering results saved to {results_file}")
    print(f"K-means: {n_clusters} clusters, silhouette score: {clustering_results['kmeans']['silhouette_score']:.4f}")
    print(f"DBSCAN: {clustering_results['dbscan']['n_clusters']} clusters, silhouette score: {clustering_results['dbscan']['silhouette_score']:.4f}")
    
    return clustering_results


def create_feature_visualization(features: np.ndarray,
                                image_paths: List[str],
                                output_dir: str,
                                cluster_labels: Optional[np.ndarray] = None):
    """
    Create 2D visualization of feature space
    
    Args:
        features: Feature matrix
        image_paths: Image paths
        output_dir: Output directory
        cluster_labels: Optional cluster labels for coloring
    """
    print("Creating feature space visualization...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy labels if not provided
    if cluster_labels is None:
        cluster_labels = np.zeros(len(features))
    
    # Create t-SNE visualization
    tsne_fig = visualize_feature_space(
        features, cluster_labels, method='tsne',
        save_path=str(output_dir / 'features_tsne.png')
    )
    
    # Create PCA visualization
    pca_fig = visualize_feature_space(
        features, cluster_labels, method='pca',
        save_path=str(output_dir / 'features_pca.png')
    )
    
    print(f"Feature visualizations saved to {output_dir}")


def analyze_feature_statistics(features: np.ndarray,
                              image_paths: List[str],
                              output_path: str):
    """
    Analyze and save feature statistics
    
    Args:
        features: Feature matrix
        image_paths: Image paths
        output_path: Output file path
    """
    print("Computing feature statistics...")
    
    stats = {
        'feature_statistics': {
            'num_images': len(image_paths),
            'feature_dim': features.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(features, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(features, axis=1))),
            'feature_mean': features.mean(axis=0).tolist(),
            'feature_std': features.std(axis=0).tolist(),
            'feature_min': features.min(axis=0).tolist(),
            'feature_max': features.max(axis=0).tolist()
        },
        'pairwise_statistics': {
            'mean_cosine_similarity': float(np.mean(np.dot(features, features.T))),
            'std_cosine_similarity': float(np.std(np.dot(features, features.T)))
        }
    }
    
    # Save statistics
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Feature statistics saved to {output_path}")
    return stats


def main():
    """Main feature extraction function"""
    args = parse_arguments()
    
    # Validate arguments
    if not args.images_dir and not args.images_list:
        raise ValueError("Must provide either --images_dir or --images_list")
    
    print("DCAL Feature Extraction")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Feature type: {args.feature_type}")
    print(f"Batch size: {args.batch_size}")
    
    # Load inference engine
    print("\nLoading model...")
    engine = TwinInferenceEngine(args.model, args.config, args.device)
    
    # Collect image paths
    print("\nCollecting image paths...")
    image_paths = collect_image_paths(
        args.images_dir, 
        args.images_list, 
        args.image_extensions
    )
    print(f"Found {len(image_paths)} valid images")
    
    if len(image_paths) == 0:
        print("No valid images found!")
        return
    
    # Extract features
    features = extract_features_batch(
        engine, image_paths, args.feature_type, args.batch_size
    )
    
    # Create output directory if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output).parent
    
    # Save main features file
    additional_info = {'model_path': args.model}
    save_features(features, image_paths, args.output, args.feature_type, additional_info)
    
    # Save individual feature files if requested
    if args.save_individual:
        individual_dir = output_dir / 'individual_features'
        save_individual_features(features, image_paths, str(individual_dir), args.feature_type)
    
    # Compute similarity matrix if requested
    similarity_matrix = None
    if args.similarity_matrix:
        similarity_path = output_dir / 'similarity_matrix.npz'
        similarity_matrix = compute_similarity_matrix(features, image_paths, str(similarity_path))
    
    # Find top-k similar images if requested
    if args.top_k_similar > 0:
        if similarity_matrix is None:
            print("Computing similarity matrix for top-k search...")
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = np.dot(features_norm, features_norm.T)
        
        top_k_path = output_dir / f'top_{args.top_k_similar}_similar.json'
        find_top_k_similar(similarity_matrix, image_paths, args.top_k_similar, str(top_k_path))
    
    # Perform clustering analysis if requested
    cluster_labels = None
    if args.cluster_features:
        clustering_dir = output_dir / 'clustering'
        clustering_results = cluster_features(features, image_paths, str(clustering_dir))
        cluster_labels = np.array(clustering_results['kmeans']['cluster_labels'])
    
    # Create feature visualizations if requested
    if args.visualize_features:
        viz_dir = output_dir / 'visualizations'
        create_feature_visualization(features, image_paths, str(viz_dir), cluster_labels)
    
    # Analyze feature statistics
    stats_path = output_dir / 'feature_statistics.json'
    analyze_feature_statistics(features, image_paths, str(stats_path))
    
    print(f"\nFeature extraction complete!")
    print(f"Main features saved to: {args.output}")
    print(f"Additional files saved to: {output_dir}")


if __name__ == '__main__':
    main() 