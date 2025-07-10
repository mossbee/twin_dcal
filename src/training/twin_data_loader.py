"""
Twin Face Verification Data Loading

This module implements optimized data loading for twin face verification:
1. TwinVerificationDataset: Dataset with maximum pair generation
2. TwinPairSampler: Smart sampling for hard negatives (twin pairs)
3. Distributed data loaders with proper synchronization
"""

import json
import random
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class TwinVerificationDataset(Dataset):
    """
    Dataset for twin face verification with maximum data utilization
    Generates balanced positive/negative pairs with twin emphasis
    """
    
    def __init__(self,
                 dataset_info_path: str,
                 twin_pairs_path: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 twin_ratio: float = 0.3,
                 max_pairs_per_epoch: Optional[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 seed: int = 42):
        super().__init__()
        
        self.split = split
        self.twin_ratio = twin_ratio
        self.transform = transform
        self.seed = seed
        
        # Load dataset information
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Validate and clean image paths
        self.dataset_info = self._validate_image_paths(self.dataset_info)
        
        # Convert twin pairs to set for faster lookup
        self.twin_pairs_set = set()
        for pair in self.twin_pairs:
            self.twin_pairs_set.add((pair[0], pair[1]))
            self.twin_pairs_set.add((pair[1], pair[0]))  # Bidirectional
        
        # Split dataset by identity (stratified)
        self.split_data(train_ratio, val_ratio)
        
        # Generate all possible pairs
        self.positive_pairs = self.generate_positive_pairs()
        self.negative_pairs = self.generate_negative_pairs()
        
        # Balance pairs and create epoch pairs
        self.max_pairs_per_epoch = max_pairs_per_epoch
        self.epoch_pairs = self.create_epoch_pairs()
        
        print(f"Dataset {split}: {len(self.identity_ids)} identities, "
              f"{len(self.positive_pairs)} positive pairs, "
              f"{len(self.negative_pairs)} negative pairs, "
              f"{len(self.epoch_pairs)} pairs per epoch")
    
    def _validate_image_paths(self, dataset_info: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate that all image paths exist and filter out missing ones"""
        validated_dataset = {}
        total_images = 0
        missing_images = 0
        
        for identity_id, image_paths in dataset_info.items():
            valid_paths = []
            for path in image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    missing_images += 1
                    print(f"Warning: Missing image {path}")
                total_images += 1
            
            # Only keep identities with at least 2 valid images (needed for positive pairs)
            if len(valid_paths) >= 2:
                validated_dataset[identity_id] = valid_paths
            else:
                print(f"Warning: Identity {identity_id} has only {len(valid_paths)} valid images, removing from dataset")
        
        if missing_images > 0:
            print(f"Dataset validation: {missing_images}/{total_images} images missing, "
                  f"kept {len(validated_dataset)} identities with valid images")
        
        return validated_dataset
    
    def split_data(self, train_ratio: float, val_ratio: float):
        """Split dataset by identity to prevent data leakage"""
        all_ids = list(self.dataset_info.keys())
        
        # Set seed for reproducible splits
        random.seed(self.seed)
        random.shuffle(all_ids)
        
        n_total = len(all_ids)
        
        # Handle external test dataset case (all data is test)
        if train_ratio == 0.0 and val_ratio == 0.0:
            # External test dataset - use all data
            if self.split == 'test':
                self.identity_ids = all_ids
            else:
                # No train/val data in external test dataset
                self.identity_ids = []
        else:
            # Regular dataset split
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)
            
            if self.split == 'train':
                self.identity_ids = all_ids[:n_train]
            elif self.split == 'val':
                self.identity_ids = all_ids[n_train:n_train+n_val]
            else:  # test
                self.identity_ids = all_ids[n_train+n_val:]
        
        # Filter dataset for current split
        self.split_dataset_info = {
            id_: paths for id_, paths in self.dataset_info.items()
            if id_ in self.identity_ids
        }
        
        # Count total images in split
        self.total_images = sum(len(paths) for paths in self.split_dataset_info.values())
    
    def generate_positive_pairs(self) -> List[Tuple[str, str, int]]:
        """Generate all possible positive pairs (same person)"""
        positive_pairs = []
        
        for identity_id, image_paths in self.split_dataset_info.items():
            if len(image_paths) >= 2:
                # Generate all combinations
                for i in range(len(image_paths)):
                    for j in range(i + 1, len(image_paths)):
                        positive_pairs.append((image_paths[i], image_paths[j], 1))
        
        return positive_pairs
    
    def generate_negative_pairs(self) -> List[Tuple[str, str, int]]:
        """Generate negative pairs with twin emphasis"""
        negative_pairs = []
        all_ids = list(self.split_dataset_info.keys())
        
        # Twin pairs (hard negatives)
        twin_negatives = []
        for id1, id2 in self.twin_pairs:
            if id1 in self.split_dataset_info and id2 in self.split_dataset_info:
                # Generate pairs between twin identities
                for img1 in self.split_dataset_info[id1]:
                    for img2 in self.split_dataset_info[id2]:
                        twin_negatives.append((img1, img2, 0))
        
        # Regular negative pairs
        regular_negatives = []
        if len(twin_negatives) > 0:
            num_regular = int(len(twin_negatives) / self.twin_ratio * (1 - self.twin_ratio))
        else:
            # If no twin pairs in this split, generate standard negatives
            num_regular = len(self.positive_pairs)
        
        attempts = 0
        max_attempts = num_regular * 10  # Prevent infinite loop
        
        while len(regular_negatives) < num_regular and attempts < max_attempts:
            attempts += 1
            
            # Sample two different identities
            id1, id2 = random.sample(all_ids, 2)
            
            # Ensure not a twin pair
            if (id1, id2) not in self.twin_pairs_set:
                img1 = random.choice(self.split_dataset_info[id1])
                img2 = random.choice(self.split_dataset_info[id2])
                regular_negatives.append((img1, img2, 0))
        
        return twin_negatives + regular_negatives
    
    def create_epoch_pairs(self) -> List[Tuple[str, str, int]]:
        """Create balanced pairs for one epoch"""
        # Balance positive and negative pairs
        min_pairs = min(len(self.positive_pairs), len(self.negative_pairs))
        
        if self.max_pairs_per_epoch is not None:
            min_pairs = min(min_pairs, self.max_pairs_per_epoch // 2)
        
        # Sample balanced pairs
        random.seed(self.seed)  # Reset for consistency
        sampled_positive = random.sample(self.positive_pairs, min_pairs)
        sampled_negative = random.sample(self.negative_pairs, min_pairs)
        
        # Combine and shuffle
        epoch_pairs = sampled_positive + sampled_negative
        random.shuffle(epoch_pairs)
        
        return epoch_pairs
    
    def __len__(self) -> int:
        return len(self.epoch_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img1_path, img2_path, label = self.epoch_pairs[idx]
        
        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images {img1_path}, {img2_path}: {e}")
            # Return a dummy sample
            img1 = Image.new('RGB', (448, 448), color='black')
            img2 = Image.new('RGB', (448, 448), color='black')
            label = 0
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': torch.tensor(label, dtype=torch.float32),
            'paths': (img1_path, img2_path),
            'idx': idx
        }
    
    def resample_epoch_pairs(self):
        """Resample pairs for new epoch (data augmentation)"""
        if self.split == 'train':  # Only resample for training
            self.epoch_pairs = self.create_epoch_pairs()


class TwinPairSampler(Sampler):
    """
    Custom sampler that ensures twin pairs are sampled more frequently
    """
    
    def __init__(self, dataset: TwinVerificationDataset, batch_size: int = 16):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        
        # Identify twin pair indices
        self.twin_indices = []
        self.regular_indices = []
        
        for idx, (img1_path, img2_path, label) in enumerate(dataset.epoch_pairs):
            if label == 0:  # Negative pair
                # Check if it's a twin pair by examining paths
                id1 = self.extract_identity_from_path(img1_path)
                id2 = self.extract_identity_from_path(img2_path)
                
                if (id1, id2) in dataset.twin_pairs_set:
                    self.twin_indices.append(idx)
                else:
                    self.regular_indices.append(idx)
            else:
                self.regular_indices.append(idx)
    
    def extract_identity_from_path(self, path: str) -> str:
        """Extract identity ID from image path (dataset-specific)"""
        # This is a placeholder - implement based on your path structure
        # For example, if paths are like "/data/id_123/image.jpg"
        parts = path.split('/')
        for part in parts:
            if part.startswith('id_'):
                return part
        return "unknown"
    
    def __iter__(self):
        # Create batches with twin pair emphasis
        indices = []
        twin_boost_factor = 2  # Sample twin pairs 2x more often
        
        # Add regular indices
        indices.extend(self.regular_indices)
        
        # Add twin indices multiple times
        for _ in range(twin_boost_factor):
            indices.extend(self.twin_indices)
        
        # Shuffle and truncate to dataset length
        random.shuffle(indices)
        indices = indices[:self.num_samples]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


def create_train_transforms(config) -> transforms.Compose:
    """Create training data transforms"""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
        # Note: No horizontal flip for faces to preserve identity
    ])


def create_val_test_transforms(config) -> transforms.Compose:
    """Create validation/test data transforms"""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])


def mixup_data(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to image pairs"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x1.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam


def create_data_loaders(config, 
                       dataset_info_path: str = None,
                       twin_pairs_path: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    Supports external test dataset for evaluation
    
    Args:
        config: Configuration object
        dataset_info_path: Path to dataset information JSON
        twin_pairs_path: Path to twin pairs JSON
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Use config paths if not provided
    if dataset_info_path is None:
        dataset_info_path = config.DATASET_INFO
    if twin_pairs_path is None:
        twin_pairs_path = config.TWIN_PAIRS_INFO
    
    # Create transforms
    train_transform = create_train_transforms(config)
    val_test_transform = create_val_test_transforms(config)
    
    # Create train and validation datasets from main dataset
    train_dataset = TwinVerificationDataset(
        dataset_info_path=dataset_info_path,
        twin_pairs_path=twin_pairs_path,
        split='train',
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        twin_ratio=config.TWIN_PAIR_RATIO,
        transform=train_transform
    )
    
    val_dataset = TwinVerificationDataset(
        dataset_info_path=dataset_info_path,
        twin_pairs_path=twin_pairs_path,
        split='val',
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        twin_ratio=config.TWIN_PAIR_RATIO,
        transform=val_test_transform
    )
    
    # Create test dataset - use external if configured, otherwise use split from main dataset
    if config.USE_EXTERNAL_TEST and config.EXTERNAL_TEST_DATASET is not None:
        print(f"Using external test dataset: {config.EXTERNAL_TEST_DATASET}")
        # Validate external test dataset exists
        if not os.path.exists(config.EXTERNAL_TEST_DATASET):
            raise FileNotFoundError(f"External test dataset not found: {config.EXTERNAL_TEST_DATASET}")
        
        # Use external test pairs if provided, otherwise use main twin pairs
        external_twin_pairs = config.EXTERNAL_TEST_PAIRS if config.EXTERNAL_TEST_PAIRS else twin_pairs_path
        
        # Validate external twin pairs exist
        if external_twin_pairs and not os.path.exists(external_twin_pairs):
            print(f"Warning: External test pairs file not found: {external_twin_pairs}")
            print("Using main twin pairs for external test dataset")
            external_twin_pairs = twin_pairs_path
        
        # Create external test dataset
        test_dataset = TwinVerificationDataset(
            dataset_info_path=config.EXTERNAL_TEST_DATASET,
            twin_pairs_path=external_twin_pairs,
            split='test',  # Use all data from external dataset as test
            train_ratio=0.0,  # All external data is test data
            val_ratio=0.0,
            twin_ratio=config.TWIN_PAIR_RATIO,
            transform=val_test_transform
        )
    else:
        # Use split from main dataset
        test_dataset = TwinVerificationDataset(
            dataset_info_path=dataset_info_path,
            twin_pairs_path=twin_pairs_path,
            split='test',
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            twin_ratio=config.TWIN_PAIR_RATIO,
            transform=val_test_transform
        )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset, 
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
            drop_last=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def collate_verification_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for verification batches"""
    img1_batch = torch.stack([item['img1'] for item in batch])
    img2_batch = torch.stack([item['img2'] for item in batch])
    labels_batch = torch.stack([item['label'] for item in batch])
    paths_batch = [item['paths'] for item in batch]
    indices_batch = torch.tensor([item['idx'] for item in batch])
    
    return {
        'img1': img1_batch,
        'img2': img2_batch,
        'labels': labels_batch,
        'paths': paths_batch,
        'indices': indices_batch
    }


# Utility functions for data loading
def calculate_dataset_stats(dataset_info_path: str) -> Dict[str, Any]:
    """Calculate statistics about the dataset"""
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    total_identities = len(dataset_info)
    total_images = sum(len(paths) for paths in dataset_info.values())
    images_per_person = [len(paths) for paths in dataset_info.values()]
    
    stats = {
        'total_identities': total_identities,
        'total_images': total_images,
        'min_images_per_person': min(images_per_person),
        'max_images_per_person': max(images_per_person),
        'avg_images_per_person': np.mean(images_per_person),
        'median_images_per_person': np.median(images_per_person)
    }
    
    return stats


def estimate_pair_counts(dataset_info_path: str, twin_pairs_path: str) -> Dict[str, int]:
    """Estimate total number of pairs that can be generated"""
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    with open(twin_pairs_path, 'r') as f:
        twin_pairs = json.load(f)
    
    # Count positive pairs
    positive_pairs = 0
    for identity_id, image_paths in dataset_info.items():
        n_images = len(image_paths)
        if n_images >= 2:
            positive_pairs += n_images * (n_images - 1) // 2
    
    # Count potential negative pairs
    total_identities = len(dataset_info)
    max_negative_pairs = 0
    for i, (id1, paths1) in enumerate(dataset_info.items()):
        for j, (id2, paths2) in enumerate(dataset_info.items()):
            if i < j:  # Avoid double counting
                max_negative_pairs += len(paths1) * len(paths2)
    
    # Count twin negative pairs
    twin_negative_pairs = 0
    for id1, id2 in twin_pairs:
        if id1 in dataset_info and id2 in dataset_info:
            twin_negative_pairs += len(dataset_info[id1]) * len(dataset_info[id2])
    
    return {
        'positive_pairs': positive_pairs,
        'max_negative_pairs': max_negative_pairs,
        'twin_negative_pairs': twin_negative_pairs,
        'regular_negative_pairs': max_negative_pairs - twin_negative_pairs
    } 