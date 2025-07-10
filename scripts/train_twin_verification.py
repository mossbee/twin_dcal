#!/usr/bin/env python3
"""
Main Training Script for Twin Face Verification using DCAL

This script handles:
1. Configuration setup and validation
2. Distributed training initialization
3. Model training orchestration
4. Logging and monitoring
5. Checkpoint management

Usage:
    # Single GPU training
    python train_twin_verification.py --config configs/single_gpu

    # Multi-GPU training (2x RTX 2080Ti)
    python train_twin_verification.py --config configs/default

    # Debug mode
    python train_twin_verification.py --config configs/debug
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist

from configs.twin_verification_config import (
    TwinVerificationConfig, 
    get_debug_config, 
    get_single_gpu_config,
    print_config_summary,
    save_config
)
from training.twin_trainer import DistributedTrainer, TwinVerificationTrainer
from training.twin_data_loader import calculate_dataset_stats, estimate_pair_counts


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DCAL for Twin Face Verification')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='default',
        choices=['default', 'debug', 'single_gpu', 'large_model'],
        help='Configuration preset to use'
    )
    
    parser.add_argument(
        '--dataset_info',
        type=str,
        default='data/dataset_infor.json',
        help='Path to dataset information JSON'
    )
    
    parser.add_argument(
        '--twin_pairs',
        type=str, 
        default='data/twin_pairs_infor.json',
        help='Path to twin pairs information JSON'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training'
    )
    
    parser.add_argument(
        '--world_size',
        type=int,
        default=None,
        help='Number of processes for distributed training'
    )
    
    parser.add_argument(
        '--master_addr',
        type=str,
        default='localhost',
        help='Master address for distributed training'
    )
    
    parser.add_argument(
        '--master_port',
        type=str,
        default='12355',
        help='Master port for distributed training'
    )
    
    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='Only run validation (requires --resume)'
    )
    
    parser.add_argument(
        '--mlflow_disabled',
        action='store_true',
        help='Disable MLFlow logging'
    )
    
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default='http://localhost:5000',
        help='MLFlow tracking server URI'
    )
    
    return parser.parse_args()


def get_config(config_name: str) -> TwinVerificationConfig:
    """Get configuration based on name"""
    if config_name == 'debug':
        return get_debug_config()
    elif config_name == 'single_gpu':
        return get_single_gpu_config()
    elif config_name == 'default':
        return TwinVerificationConfig()
    else:
        raise ValueError(f"Unknown config: {config_name}")


def setup_environment(args, config):
    """Setup training environment"""
    # Set environment variables for distributed training
    if args.world_size is not None:
        config.WORLD_SIZE = args.world_size
    
    if args.master_addr != 'localhost':
        config.MASTER_ADDR = args.master_addr
    
    if args.master_port != '12355':
        config.MASTER_PORT = args.master_port
    
    # Override paths if provided
    if args.dataset_info:
        config.DATASET_INFO = args.dataset_info
    
    if args.twin_pairs:
        config.TWIN_PAIRS_INFO = args.twin_pairs
    
    # Set output directory
    if args.output_dir:
        config.SAVE_DIR = args.output_dir
    
    # Set MLFlow URI if provided
    if args.mlflow_uri:
        config.MLFLOW_TRACKING_URI = args.mlflow_uri
    
    # Disable MLFlow if requested
    if args.mlflow_disabled:
        config.MLFLOW_EXPERIMENT_NAME = None
    
    # Validate configuration
    config.validate_config()
    
    # Create output directory
    output_dir = config.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    save_config(config, config_path)
    
    return config


def print_dataset_info(config):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    try:
        # Calculate dataset statistics
        stats = calculate_dataset_stats(config.DATASET_INFO)
        print(f"Total identities: {stats['total_identities']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Images per person: {stats['min_images_per_person']}-{stats['max_images_per_person']} "
              f"(avg: {stats['avg_images_per_person']:.1f})")
        
        # Estimate pair counts
        pair_counts = estimate_pair_counts(config.DATASET_INFO, config.TWIN_PAIRS_INFO)
        print(f"Positive pairs: {pair_counts['positive_pairs']:,}")
        print(f"Twin negative pairs: {pair_counts['twin_negative_pairs']:,}")
        print(f"Regular negative pairs: {pair_counts['regular_negative_pairs']:,}")
        
    except Exception as e:
        print(f"Warning: Could not load dataset info: {e}")
    
    print("="*60)


def single_gpu_main(rank, args):
    """Main function for single GPU training"""
    # Get configuration
    config = get_config(args.config)
    config = setup_environment(args, config)
    
    # Print information
    if rank == 0:
        print_config_summary(config)
        print_dataset_info(config)
    
    # Create trainer (no distributed trainer for single GPU)
    trainer = TwinVerificationTrainer(config, distributed_trainer=None)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training or validation
    try:
        if args.validate_only:
            if not args.resume:
                raise ValueError("--validate_only requires --resume")
            print("Running validation only...")
            val_metrics = trainer.validate()
            print(f"Validation results: {val_metrics}")
        else:
            trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


def distributed_main(rank, world_size, args):
    """Main function for distributed training"""
    # Setup distributed training
    distributed_trainer = DistributedTrainer(get_config(args.config))
    distributed_trainer.setup(rank, world_size)
    
    try:
        # Get configuration
        config = get_config(args.config)
        config = setup_environment(args, config)
        
        # Print information (only on main process)
        if distributed_trainer.is_main_process():
            print_config_summary(config)
            print_dataset_info(config)
        
        # Create trainer
        trainer = TwinVerificationTrainer(config, distributed_trainer)
        
        # Resume from checkpoint if provided
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Synchronize all processes
        distributed_trainer.barrier()
        
        # Run training or validation
        try:
            if args.validate_only:
                if not args.resume:
                    raise ValueError("--validate_only requires --resume")
                if distributed_trainer.is_main_process():
                    print("Running validation only...")
                val_metrics = trainer.validate()
                if distributed_trainer.is_main_process():
                    print(f"Validation results: {val_metrics}")
            else:
                trainer.train()
        except KeyboardInterrupt:
            if distributed_trainer.is_main_process():
                print("Training interrupted by user")
        except Exception as e:
            if distributed_trainer.is_main_process():
                print(f"Training failed: {e}")
            raise
        finally:
            trainer.cleanup()
    
    finally:
        distributed_trainer.cleanup()


def check_requirements():
    """Check if all requirements are satisfied"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow on CPU.")
        return False
    
    # Check GPU memory
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f"  GPU {i}: {gpu_props.name}, {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print(f"Warning: GPU {i} has less than 8GB memory. Consider reducing batch size.")
    
    return True


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Check requirements
    if not check_requirements():
        print("Continuing despite warnings...")
    
    # Determine world size
    if args.local_rank != -1:
        # Launched with torch.distributed.launch
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
    elif args.world_size is not None:
        # Manual specification
        world_size = args.world_size
        rank = 0  # Will be overridden in spawn
    else:
        # Auto-detect based on available GPUs
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        rank = 0
    
    print(f"Training setup: world_size={world_size}, local_rank={args.local_rank}")
    
    # Choose training mode
    if world_size == 1:
        # Single GPU or CPU training
        print("Starting single GPU/CPU training...")
        single_gpu_main(0, args)
    else:
        # Distributed training
        print("Starting distributed training...")
        if args.local_rank != -1:
            # Already in distributed launch context
            distributed_main(rank, world_size, args)
        else:
            # Use mp.spawn to launch distributed training
            mp.spawn(
                distributed_main,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )


if __name__ == '__main__':
    main() 