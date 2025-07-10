"""
Distributed Training Utilities for Twin Face Verification

This module provides utilities for distributed training:
1. Process group setup and cleanup
2. Tensor synchronization across processes
3. Distributed logging and monitoring
4. Multi-GPU memory management
"""

import os
import sys
import time
import socket
import pickle
from typing import Dict, Any, Optional, List, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed(rank: int, 
                     world_size: int,
                     master_addr: str = 'localhost',
                     master_port: Optional[str] = None,
                     backend: str = 'nccl') -> bool:
    """
    Setup distributed training process group
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend ('nccl', 'gloo')
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        
        if master_port is None:
            master_port = str(find_free_port())
        os.environ['MASTER_PORT'] = master_port
        
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        
        # Set CUDA device
        if torch.cuda.is_available() and backend == 'nccl':
            local_rank = rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
        
        # Test communication
        if dist.is_initialized():
            test_tensor = torch.tensor([rank], dtype=torch.float32)
            if torch.cuda.is_available() and backend == 'nccl':
                test_tensor = test_tensor.cuda()
            
            dist.all_reduce(test_tensor)
            
            if is_main_process():
                print(f"Distributed training setup successful:")
                print(f"  Backend: {backend}")
                print(f"  World size: {world_size}")
                print(f"  Master: {master_addr}:{master_port}")
                print(f"  Available GPUs: {torch.cuda.device_count()}")
        
        return True
        
    except Exception as e:
        print(f"Failed to setup distributed training: {e}")
        return False


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank within node"""
    if not dist.is_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def barrier():
    """Synchronize all processes"""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, 
                 op: str = 'mean',
                 group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'mean', 'max', 'min')
        group: Process group (None for default)
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    # Clone tensor to avoid modifying original
    reduced_tensor = tensor.clone()
    
    # Perform reduction
    if op == 'sum':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM, group=group)
    elif op == 'mean':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM, group=group)
        reduced_tensor /= world_size
    elif op == 'max':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MAX, group=group)
    elif op == 'min':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MIN, group=group)
    else:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    return reduced_tensor


def all_gather_tensor(tensor: torch.Tensor,
                     group: Optional[dist.ProcessGroup] = None) -> List[torch.Tensor]:
    """
    Gather tensor from all processes
    
    Args:
        tensor: Tensor to gather
        group: Process group (None for default)
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    # Create list to store tensors from all processes
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather tensors
    dist.all_gather(tensor_list, tensor, group=group)
    
    return tensor_list


def broadcast_tensor(tensor: torch.Tensor,
                    src: int = 0,
                    group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """
    Broadcast tensor from source process to all processes
    
    Args:
        tensor: Tensor to broadcast
        src: Source process rank
        group: Process group (None for default)
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def all_gather_object(obj: Any,
                     group: Optional[dist.ProcessGroup] = None) -> List[Any]:
    """
    Gather object from all processes (works with any picklable object)
    
    Args:
        obj: Object to gather
        group: Process group (None for default)
        
    Returns:
        List of objects from all processes
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    
    # Create list to store objects from all processes
    object_list = [None] * world_size
    
    # Gather objects
    dist.all_gather_object(object_list, obj, group=group)
    
    return object_list


def synchronize_metrics(metrics: Dict[str, float],
                       group: Optional[dist.ProcessGroup] = None) -> Dict[str, float]:
    """
    Synchronize metrics across all processes
    
    Args:
        metrics: Dictionary of metrics to synchronize
        group: Process group (None for default)
        
    Returns:
        Synchronized metrics (averaged across processes)
    """
    if not dist.is_initialized():
        return metrics
    
    synchronized_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Convert to tensor and reduce
            value_tensor = torch.tensor(float(value))
            if torch.cuda.is_available():
                value_tensor = value_tensor.cuda()
            
            reduced_value = reduce_tensor(value_tensor, op='mean', group=group)
            synchronized_metrics[key] = reduced_value.item()
        else:
            # Keep non-numeric values as is
            synchronized_metrics[key] = value
    
    return synchronized_metrics


class DistributedLogger:
    """
    Logger that handles distributed training
    Only logs from main process to avoid duplicate logs
    """
    
    def __init__(self, name: str = "DistributedLogger"):
        self.name = name
        self.is_main = is_main_process()
        self.rank = get_rank()
        self.world_size = get_world_size()
    
    def info(self, message: str, all_ranks: bool = False):
        """Log info message"""
        if self.is_main or all_ranks:
            prefix = f"[Rank {self.rank}]" if all_ranks else ""
            print(f"{prefix} {message}")
    
    def warning(self, message: str, all_ranks: bool = False):
        """Log warning message"""
        if self.is_main or all_ranks:
            prefix = f"[Rank {self.rank}]" if all_ranks else ""
            print(f"{prefix} WARNING: {message}")
    
    def error(self, message: str, all_ranks: bool = True):
        """Log error message"""
        if self.is_main or all_ranks:
            prefix = f"[Rank {self.rank}]" if all_ranks else ""
            print(f"{prefix} ERROR: {message}")
    
    def debug(self, message: str, all_ranks: bool = False):
        """Log debug message"""
        if self.is_main or all_ranks:
            prefix = f"[Rank {self.rank}]" if all_ranks else ""
            print(f"{prefix} DEBUG: {message}")


class DistributedMemoryManager:
    """
    Memory management utilities for distributed training
    """
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information for current process"""
        if not torch.cuda.is_available():
            return {'device': 'cpu', 'memory_usage': 'N/A'}
        
        device_id = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
        
        return {
            'device_id': device_id,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'memory_total_gb': memory_total,
            'memory_usage_percent': (memory_allocated / memory_total) * 100
        }
    
    @staticmethod
    def collect_memory_stats() -> Dict[str, Any]:
        """Collect memory statistics from all processes"""
        local_stats = DistributedMemoryManager.get_memory_info()
        
        if not dist.is_initialized():
            return {'rank_0': local_stats}
        
        # Gather from all processes
        all_stats = all_gather_object(local_stats)
        
        # Organize by rank
        stats_by_rank = {f'rank_{i}': stats for i, stats in enumerate(all_stats)}
        
        # Calculate summary statistics
        if torch.cuda.is_available():
            allocated_list = [stats['memory_allocated_gb'] for stats in all_stats if 'memory_allocated_gb' in stats]
            if allocated_list:
                stats_by_rank['summary'] = {
                    'total_allocated_gb': sum(allocated_list),
                    'avg_allocated_gb': np.mean(allocated_list),
                    'max_allocated_gb': max(allocated_list),
                    'min_allocated_gb': min(allocated_list)
                }
        
        return stats_by_rank
    
    @staticmethod
    def log_memory_stats(logger: Optional[DistributedLogger] = None):
        """Log memory statistics"""
        if logger is None:
            logger = DistributedLogger()
        
        stats = DistributedMemoryManager.collect_memory_stats()
        
        if is_main_process():
            logger.info("=== Memory Statistics ===")
            for rank, rank_stats in stats.items():
                if rank.startswith('rank_'):
                    if 'memory_allocated_gb' in rank_stats:
                        logger.info(f"{rank}: {rank_stats['memory_allocated_gb']:.2f}GB allocated "
                                   f"({rank_stats['memory_usage_percent']:.1f}% of {rank_stats['memory_total_gb']:.1f}GB)")
                    else:
                        logger.info(f"{rank}: {rank_stats}")
            
            if 'summary' in stats:
                summary = stats['summary']
                logger.info(f"Summary: Total={summary['total_allocated_gb']:.2f}GB, "
                           f"Avg={summary['avg_allocated_gb']:.2f}GB, "
                           f"Max={summary['max_allocated_gb']:.2f}GB")
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache on all processes"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_ddp_model(model: torch.nn.Module,
                    device_ids: Optional[List[int]] = None,
                    output_device: Optional[int] = None,
                    find_unused_parameters: bool = False) -> DDP:
    """
    Create DistributedDataParallel model
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs
        output_device: Output device ID
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP wrapped model
    """
    if not dist.is_initialized():
        return model
    
    local_rank = get_local_rank()
    
    if device_ids is None:
        device_ids = [local_rank]
    
    if output_device is None:
        output_device = local_rank
    
    # Move model to correct device
    if torch.cuda.is_available():
        model = model.cuda(local_rank)
    
    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters
    )
    
    return ddp_model


def save_checkpoint_distributed(state_dict: Dict[str, Any],
                               checkpoint_path: str,
                               is_best: bool = False,
                               best_path: Optional[str] = None):
    """
    Save checkpoint only from main process
    
    Args:
        state_dict: State dictionary to save
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        best_path: Path to save best checkpoint
    """
    if is_main_process():
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        
        # Save best checkpoint if needed
        if is_best and best_path:
            torch.save(state_dict, best_path)


def load_checkpoint_distributed(checkpoint_path: str,
                               map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Load checkpoint with proper device mapping for distributed training
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        
    Returns:
        Loaded checkpoint state dictionary
    """
    if map_location is None:
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            map_location = f'cuda:{local_rank}'
        else:
            map_location = 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    return checkpoint


def init_seeds(seed: int = 42, deterministic: bool = False):
    """
    Initialize random seeds for reproducible distributed training
    
    Args:
        seed: Base random seed
        deterministic: Whether to use deterministic algorithms
    """
    rank = get_rank()
    
    # Set different seeds for different processes to ensure data diversity
    process_seed = seed + rank
    
    # Set Python random seed
    import random
    random.seed(process_seed)
    
    # Set NumPy random seed
    np.random.seed(process_seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)
    
    # Set deterministic behavior (may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)


def print_distributed_info():
    """Print information about distributed setup"""
    if not dist.is_initialized():
        print("Distributed training not initialized")
        return
    
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    print(f"Distributed Info - Rank: {rank}/{world_size}, Local Rank: {local_rank}")
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        print(f"CUDA Device: {device} ({device_name})")
    else:
        print("Using CPU")


# Convenience function for common setup
def setup_for_distributed_training(rank: int,
                                  world_size: int,
                                  master_addr: str = 'localhost',
                                  master_port: Optional[str] = None,
                                  seed: int = 42) -> DistributedLogger:
    """
    Complete setup for distributed training
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        seed: Random seed
        
    Returns:
        Distributed logger
    """
    # Setup distributed training
    success = setup_distributed(rank, world_size, master_addr, master_port)
    
    if not success:
        raise RuntimeError("Failed to setup distributed training")
    
    # Initialize seeds
    init_seeds(seed)
    
    # Create logger
    logger = DistributedLogger("DistributedTraining")
    
    # Print info
    if is_main_process():
        print_distributed_info()
        logger.info(f"Distributed training setup complete with {world_size} processes")
    
    return logger 