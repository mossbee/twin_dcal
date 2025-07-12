#!/usr/bin/env python3
"""
Data Loading Optimization Script for Twin Face Verification

This script optimizes training data loading by:
1. Pre-generating all image pairs to avoid runtime computation
2. Creating cached preprocessed image tensors
3. Optimizing data loading pipeline
"""

import os
import json
import time
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchvision.transforms as transforms
from PIL import Image

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from configs.twin_verification_config import TwinVerificationConfig


def create_optimized_pairs(dataset_info_path: str, 
                          twin_pairs_path: str, 
                          output_path: str,
                          train_ratio: float = 0.9,
                          val_ratio: float = 0.1,
                          twin_ratio: float = 0.3,
                          seed: int = 42) -> None:
    """
    Pre-generate all training pairs to avoid runtime computation
    
    Args:
        dataset_info_path: Path to dataset info JSON
        twin_pairs_path: Path to twin pairs JSON
        output_path: Path to save optimized pairs
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        twin_ratio: Ratio of twin pairs in negatives
        seed: Random seed for reproducibility
    """
    print("Creating optimized pairs...")
    
    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    with open(twin_pairs_path, 'r') as f:
        twin_pairs = json.load(f)
    
    # Convert twin pairs to set for faster lookup
    twin_pairs_set = set()
    for pair in twin_pairs:
        twin_pairs_set.add((pair[0], pair[1]))
        twin_pairs_set.add((pair[1], pair[0]))
    
    # Split identities
    import random
    random.seed(seed)
    all_ids = list(dataset_info.keys())
    random.shuffle(all_ids)
    
    n_total = len(all_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train+n_val]
    test_ids = all_ids[n_train+n_val:]
    
    # Generate pairs for each split
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    optimized_pairs = {}
    
    for split, identity_ids in splits.items():
        print(f"Generating {split} pairs...")
        
        # Filter dataset for current split
        split_dataset_info = {
            id_: paths for id_, paths in dataset_info.items()
            if id_ in identity_ids
        }
        
        # Generate positive pairs
        positive_pairs = []
        for identity_id, image_paths in split_dataset_info.items():
            if len(image_paths) >= 2:
                for i in range(len(image_paths)):
                    for j in range(i + 1, len(image_paths)):
                        positive_pairs.append((image_paths[i], image_paths[j], 1))
        
        # Generate negative pairs
        negative_pairs = []
        
        # Twin pairs (hard negatives)
        twin_negatives = []
        for id1, id2 in twin_pairs:
            if id1 in split_dataset_info and id2 in split_dataset_info:
                for img1 in split_dataset_info[id1]:
                    for img2 in split_dataset_info[id2]:
                        twin_negatives.append((img1, img2, 0))
        
        # Regular negative pairs
        regular_negatives = []
        if len(twin_negatives) > 0:
            num_regular = int(len(twin_negatives) / twin_ratio * (1 - twin_ratio))
        else:
            num_regular = len(positive_pairs)
        
        attempts = 0
        max_attempts = num_regular * 10
        
        while len(regular_negatives) < num_regular and attempts < max_attempts:
            attempts += 1
            
            if len(identity_ids) >= 2:
                id1, id2 = random.sample(identity_ids, 2)
                
                if (id1, id2) not in twin_pairs_set:
                    img1 = random.choice(split_dataset_info[id1])
                    img2 = random.choice(split_dataset_info[id2])
                    regular_negatives.append((img1, img2, 0))
        
        negative_pairs = twin_negatives + regular_negatives
        
        # Combine and balance
        min_pairs = min(len(positive_pairs), len(negative_pairs))
        sampled_positive = random.sample(positive_pairs, min_pairs)
        sampled_negative = random.sample(negative_pairs, min_pairs)
        
        all_pairs = sampled_positive + sampled_negative
        random.shuffle(all_pairs)
        
        optimized_pairs[split] = {
            'pairs': all_pairs,
            'positive_count': len(sampled_positive),
            'negative_count': len(sampled_negative),
            'twin_negative_count': len(twin_negatives),
            'regular_negative_count': len(regular_negatives)
        }
        
        print(f"{split}: {len(all_pairs)} total pairs ({len(sampled_positive)} pos, {len(sampled_negative)} neg)")
    
    # Save optimized pairs
    with open(output_path, 'wb') as f:
        pickle.dump(optimized_pairs, f)
    
    print(f"Optimized pairs saved to {output_path}")


def preprocess_image(image_path: str, transform: transforms.Compose) -> Optional[torch.Tensor]:
    """Preprocess a single image"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return transform(img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def create_cached_tensors(optimized_pairs_path: str,
                         output_dir: str,
                         config: TwinVerificationConfig,
                         num_workers: int = 8) -> None:
    """
    Create cached preprocessed tensors for faster data loading
    
    Args:
        optimized_pairs_path: Path to optimized pairs pickle file
        output_dir: Directory to save cached tensors
        config: Configuration object
        num_workers: Number of worker threads for preprocessing
    """
    print("Creating cached tensors...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load optimized pairs
    with open(optimized_pairs_path, 'rb') as f:
        optimized_pairs = pickle.load(f)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # Collect all unique image paths
    all_image_paths = set()
    for split_data in optimized_pairs.values():
        for img1_path, img2_path, _ in split_data['pairs']:
            all_image_paths.add(img1_path)
            all_image_paths.add(img2_path)
    
    all_image_paths = list(all_image_paths)
    print(f"Preprocessing {len(all_image_paths)} unique images...")
    
    # Create cached tensors using thread pool
    cached_tensors = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(preprocess_image, path, transform): path
            for path in all_image_paths
        }
        
        # Process results
        for future in tqdm(as_completed(future_to_path), total=len(all_image_paths), desc="Preprocessing"):
            path = future_to_path[future]
            try:
                tensor = future.result()
                if tensor is not None:
                    cached_tensors[path] = tensor
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    # Save cached tensors
    cache_path = os.path.join(output_dir, 'cached_tensors.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_tensors, f)
    
    print(f"Cached tensors saved to {cache_path}")
    print(f"Successfully cached {len(cached_tensors)} images")


def create_training_batches(optimized_pairs_path: str,
                           cached_tensors_path: str,
                           output_dir: str,
                           batch_size: int = 16) -> None:
    """
    Pre-create training batches for even faster loading
    
    Args:
        optimized_pairs_path: Path to optimized pairs
        cached_tensors_path: Path to cached tensors
        output_dir: Directory to save batched data
        batch_size: Batch size for pre-batching
    """
    print("Creating pre-batched training data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(optimized_pairs_path, 'rb') as f:
        optimized_pairs = pickle.load(f)
    
    with open(cached_tensors_path, 'rb') as f:
        cached_tensors = pickle.load(f)
    
    # Create batches for each split
    for split, split_data in optimized_pairs.items():
        print(f"Creating {split} batches...")
        
        pairs = split_data['pairs']
        num_batches = (len(pairs) + batch_size - 1) // batch_size
        
        split_batches = []
        
        for batch_idx in tqdm(range(num_batches), desc=f"Batching {split}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pairs))
            batch_pairs = pairs[start_idx:end_idx]
            
            batch_img1 = []
            batch_img2 = []
            batch_labels = []
            
            for img1_path, img2_path, label in batch_pairs:
                if img1_path in cached_tensors and img2_path in cached_tensors:
                    batch_img1.append(cached_tensors[img1_path])
                    batch_img2.append(cached_tensors[img2_path])
                    batch_labels.append(label)
            
            if batch_img1:  # Only add batch if it has valid data
                batch_data = {
                    'img1': torch.stack(batch_img1),
                    'img2': torch.stack(batch_img2),
                    'labels': torch.tensor(batch_labels, dtype=torch.float32)
                }
                split_batches.append(batch_data)
        
        # Save batches
        batch_path = os.path.join(output_dir, f'{split}_batches.pkl')
        with open(batch_path, 'wb') as f:
            pickle.dump(split_batches, f)
        
        print(f"{split}: {len(split_batches)} batches saved to {batch_path}")


def benchmark_data_loading(original_config: TwinVerificationConfig,
                          optimized_data_dir: str,
                          num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark data loading performance
    
    Args:
        original_config: Configuration for original data loading
        optimized_data_dir: Directory with optimized data
        num_iterations: Number of iterations to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    print("Benchmarking data loading performance...")
    
    # Benchmark original data loading
    print("Testing original data loading...")
    from src.training.twin_data_loader import create_data_loaders
    
    start_time = time.time()
    train_loader, _, _ = create_data_loaders(original_config)
    
    original_times = []
    for i, batch in enumerate(train_loader):
        if i >= num_iterations:
            break
        batch_start = time.time()
        # Simulate processing
        _ = batch['img1'], batch['img2'], batch['labels']
        batch_end = time.time()
        original_times.append(batch_end - batch_start)
    
    original_avg_time = np.mean(original_times)
    
    # Benchmark optimized data loading
    print("Testing optimized data loading...")
    batches_path = os.path.join(optimized_data_dir, 'train_batches.pkl')
    
    if os.path.exists(batches_path):
        with open(batches_path, 'rb') as f:
            train_batches = pickle.load(f)
        
        optimized_times = []
        for i in range(min(num_iterations, len(train_batches))):
            batch_start = time.time()
            batch = train_batches[i]
            # Simulate processing
            _ = batch['img1'], batch['img2'], batch['labels']
            batch_end = time.time()
            optimized_times.append(batch_end - batch_start)
        
        optimized_avg_time = np.mean(optimized_times)
        speedup = original_avg_time / optimized_avg_time
        
        results = {
            'original_avg_time': original_avg_time,
            'optimized_avg_time': optimized_avg_time,
            'speedup': speedup
        }
        
        print(f"Original average time: {original_avg_time:.4f}s")
        print(f"Optimized average time: {optimized_avg_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        return results
    else:
        print("Optimized batches not found, skipping benchmark")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Optimize data loading for twin face verification")
    parser.add_argument('--dataset_info', type=str, required=True,
                       help='Path to dataset info JSON')
    parser.add_argument('--twin_pairs', type=str, required=True,
                       help='Path to twin pairs JSON')
    parser.add_argument('--output_dir', type=str, default='optimized_data',
                       help='Output directory for optimized data')
    parser.add_argument('--config', type=str, default='kaggle_p100_fast',
                       help='Configuration to use')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker threads for preprocessing')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for pre-batching')
    parser.add_argument('--skip_pairs', action='store_true',
                       help='Skip pair generation step')
    parser.add_argument('--skip_tensors', action='store_true',
                       help='Skip tensor caching step')
    parser.add_argument('--skip_batches', action='store_true',
                       help='Skip batch creation step')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get configuration
    if args.config == 'kaggle_p100_fast':
        from configs.twin_verification_config import get_kaggle_p100_fast_config
        config = get_kaggle_p100_fast_config()
    else:
        config = TwinVerificationConfig()
    
    # Step 1: Create optimized pairs
    pairs_path = os.path.join(args.output_dir, 'optimized_pairs.pkl')
    if not args.skip_pairs:
        create_optimized_pairs(
            dataset_info_path=args.dataset_info,
            twin_pairs_path=args.twin_pairs,
            output_path=pairs_path,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            twin_ratio=config.TWIN_PAIR_RATIO
        )
    
    # Step 2: Create cached tensors
    tensors_path = os.path.join(args.output_dir, 'cached_tensors.pkl')
    if not args.skip_tensors:
        create_cached_tensors(
            optimized_pairs_path=pairs_path,
            output_dir=args.output_dir,
            config=config,
            num_workers=args.num_workers
        )
    
    # Step 3: Create pre-batched data
    if not args.skip_batches:
        create_training_batches(
            optimized_pairs_path=pairs_path,
            cached_tensors_path=tensors_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    
    # Step 4: Benchmark if requested
    if args.benchmark:
        benchmark_data_loading(
            original_config=config,
            optimized_data_dir=args.output_dir,
            num_iterations=50
        )
    
    print("Data optimization complete!")


if __name__ == "__main__":
    main() 