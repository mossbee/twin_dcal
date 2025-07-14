#!/usr/bin/env python3
"""
Automated Configuration Search for Twin Face Verification

This script automatically tests multiple configurations from best performance 
to lowest memory usage, finding the optimal configuration that fully utilizes 
available GPU memory without OOM errors.

Usage:
    # Test P100 single GPU configurations
    python scripts/automated_config_search.py --gpu_type p100 --test_duration 300

    # Test T4 distributed configurations  
    python scripts/automated_config_search.py --gpu_type t4_distributed --test_duration 300
"""

import os
import sys
import time
import traceback
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

import torch
import torch.cuda
import psutil
import GPUtil

from configs.twin_verification_config import (
    generate_config_search_space,
    generate_distributed_t4_search_space,
    config_dict_to_object,
    print_config_summary
)
from training.twin_trainer import TwinVerificationTrainer, DistributedTrainer


class ConfigSearchResults:
    """Class to store and analyze config search results"""
    
    def __init__(self):
        self.results = []
        self.successful_configs = []
        self.failed_configs = []
        self.best_config = None
        self.best_score = 0.0
    
    def add_result(self, config_dict: Dict[str, Any], result: Dict[str, Any]):
        """Add a configuration test result"""
        result_entry = {
            'config': config_dict,
            'result': result,
            'timestamp': time.time()
        }
        self.results.append(result_entry)
        
        if result['success']:
            self.successful_configs.append(result_entry)
            # Calculate composite score (higher is better)
            score = self._calculate_performance_score(result)
            if score > self.best_score:
                self.best_score = score
                self.best_config = result_entry
        else:
            self.failed_configs.append(result_entry)
    
    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """Calculate performance score for ranking configs"""
        # Higher scores = better performance
        score = 0.0
        
        # Memory utilization (prefer higher utilization)
        memory_util = result.get('peak_memory_gb', 0) / result.get('total_memory_gb', 16)
        score += memory_util * 40  # 40 points max for memory utilization
        
        # Training speed (prefer faster)
        samples_per_sec = result.get('samples_per_second', 0)
        score += min(samples_per_sec * 2, 30)  # 30 points max for speed
        
        # Model complexity (prefer higher complexity if it works)
        pwca_blocks = result['config'].get('PWCA_BLOCKS', 0)
        sa_blocks = result['config'].get('SA_BLOCKS', 0)
        input_size = result['config'].get('INPUT_SIZE', 224)
        d_model = result['config'].get('D_MODEL', 384)
        
        score += pwca_blocks * 2  # 2 points per PWCA block
        score += sa_blocks * 1    # 1 point per SA block
        score += (input_size / 224) * 10  # 10 points for 448x448 vs 224x224
        score += (d_model / 384) * 10     # 10 points for 768 vs 384 dims
        
        return score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of search results"""
        total = len(self.results)
        successful = len(self.successful_configs)
        failed = len(self.failed_configs)
        
        return {
            'total_configs_tested': total,
            'successful_configs': successful,
            'failed_configs': failed,
            'success_rate': successful / total if total > 0 else 0.0,
            'best_config': self.best_config['config'] if self.best_config else None,
            'best_score': self.best_score,
            'all_results': self.results
        }


class ConfigTester:
    """Class to test individual configurations"""
    
    def __init__(self, test_duration: int = 300):
        self.test_duration = test_duration  # seconds
        self.results = ConfigSearchResults()
    
    def test_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single configuration"""
        config_name = config_dict.get('name', 'unnamed')
        description = config_dict.get('description', 'No description')
        
        print(f"\n{'='*60}")
        print(f"Testing Config: {config_name}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        # Convert to config object
        try:
            config = config_dict_to_object(config_dict)
            print_config_summary(config)
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return {
                'success': False,
                'error': f"Config validation failed: {e}",
                'config': config_dict
            }
        
        # Test the configuration
        result = self._run_training_test(config, config_name)
        
        # Add to results
        self.results.add_result(config_dict, result)
        
        return result
    
    def _run_training_test(self, config, config_name: str) -> Dict[str, Any]:
        """Run a short training test with the given configuration"""
        start_time = time.time()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get initial memory stats
        initial_memory = self._get_memory_stats()
        
        try:
            # Create trainer
            distributed_trainer = None
            if config.WORLD_SIZE > 1:
                distributed_trainer = DistributedTrainer(config)
                distributed_trainer.setup(0, config.WORLD_SIZE)
            
            trainer = TwinVerificationTrainer(config, distributed_trainer)
            
            # Get memory after model creation
            model_memory = self._get_memory_stats()
            
            # Run training for limited time
            success = self._run_limited_training(trainer, config_name)
            
            # Get peak memory usage
            peak_memory = self._get_memory_stats()
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Cleanup
            trainer.cleanup()
            if distributed_trainer:
                distributed_trainer.cleanup()
            
            torch.cuda.empty_cache()
            
            result = {
                'success': success,
                'duration': duration,
                'initial_memory_gb': initial_memory['used_gb'],
                'model_memory_gb': model_memory['used_gb'],
                'peak_memory_gb': peak_memory['used_gb'],
                'total_memory_gb': peak_memory['total_gb'],
                'memory_utilization': peak_memory['used_gb'] / peak_memory['total_gb'],
                'config': config.to_dict(),
                'samples_per_second': 0.0,  # Will be calculated during training
                'error': None
            }
            
            if success:
                print(f"✅ Config '{config_name}' SUCCESS")
                print(f"   Memory: {peak_memory['used_gb']:.1f}GB / {peak_memory['total_gb']:.1f}GB ({peak_memory['used_gb']/peak_memory['total_gb']*100:.1f}%)")
                print(f"   Duration: {duration:.1f}s")
            else:
                print(f"⚠️  Config '{config_name}' TIMEOUT (may work with longer training)")
                
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ Config '{config_name}' FAILED: Out of Memory")
            print(f"   Error: {str(e)}")
            return {
                'success': False,
                'error': 'OutOfMemoryError',
                'error_details': str(e),
                'config': config.to_dict(),
                'duration': time.time() - start_time,
                'peak_memory_gb': self._get_memory_stats()['used_gb'],
                'total_memory_gb': self._get_memory_stats()['total_gb'],
            }
            
        except Exception as e:
            print(f"❌ Config '{config_name}' FAILED: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            return {
                'success': False,
                'error': type(e).__name__,
                'error_details': str(e),
                'config': config.to_dict(),
                'duration': time.time() - start_time,
                'traceback': traceback.format_exc(),
            }
    
    def _run_limited_training(self, trainer, config_name: str) -> bool:
        """Run training for a limited time and return success status"""
        start_time = time.time()
        
        try:
            # Start training in a separate process to handle timeouts
            epoch_start = time.time()
            
            # Run a few training steps to test memory and speed
            for epoch in range(min(2, trainer.config.EPOCHS)):
                trainer.epoch = epoch
                
                # Test a few batches
                batch_count = 0
                for batch_idx, batch in enumerate(trainer.train_loader):
                    if time.time() - start_time > self.test_duration:
                        print(f"⏰ Timeout reached for config '{config_name}' after {self.test_duration}s")
                        return False
                    
                    # Move data to device
                    img1 = batch['img1'].to(trainer.device, non_blocking=True)
                    img2 = batch['img2'].to(trainer.device, non_blocking=True)
                    labels = batch['labels'].to(trainer.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = trainer.model(img1, img2, training=True, return_features=True)
                    loss, _ = trainer.loss_fn(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()
                    
                    batch_count += 1
                    
                    # Test enough batches to ensure stability
                    if batch_count >= 20:  # Test 20 batches
                        break
                
                # Calculate samples per second
                epoch_time = time.time() - epoch_start
                samples_processed = batch_count * trainer.config.BATCH_SIZE_PER_GPU
                samples_per_second = samples_processed / epoch_time if epoch_time > 0 else 0
                
                print(f"   Epoch {epoch}: {batch_count} batches, {samples_per_second:.1f} samples/s")
                
                epoch_start = time.time()
            
            return True
            
        except Exception as e:
            print(f"   Training failed: {e}")
            return False
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            return {
                'total_gb': gpu_memory,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'used_gb': max(allocated, cached)
            }
        return {'total_gb': 0, 'allocated_gb': 0, 'cached_gb': 0, 'used_gb': 0}
    
    def run_search(self, config_search_space: List[Dict[str, Any]]) -> ConfigSearchResults:
        """Run the complete configuration search"""
        print(f"\n🔍 Starting automated configuration search...")
        print(f"   Total configurations to test: {len(config_search_space)}")
        print(f"   Test duration per config: {self.test_duration}s")
        
        for i, config_dict in enumerate(config_search_space):
            print(f"\n📊 Progress: {i+1}/{len(config_search_space)}")
            result = self.test_config(config_dict)
            
            # Early termination if we found a good working config
            if result['success'] and len(self.results.successful_configs) >= 3:
                print(f"\n🎯 Found {len(self.results.successful_configs)} working configurations, stopping search.")
                break
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print search results summary"""
        summary = self.results.get_summary()
        
        print(f"\n{'='*60}")
        print(f"CONFIGURATION SEARCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total configs tested: {summary['total_configs_tested']}")
        print(f"Successful configs: {summary['successful_configs']}")
        print(f"Failed configs: {summary['failed_configs']}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        
        if summary['best_config']:
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Name: {summary['best_config']['name']}")
            print(f"   Score: {summary['best_score']:.1f}")
            print(f"   Input size: {summary['best_config']['INPUT_SIZE']}")
            print(f"   PWCA blocks: {summary['best_config']['PWCA_BLOCKS']}")
            print(f"   SA blocks: {summary['best_config']['SA_BLOCKS']}")
            print(f"   Batch size: {summary['best_config']['BATCH_SIZE_PER_GPU']}")
            print(f"   Model dims: {summary['best_config']['D_MODEL']}")
            
            # Save best config
            best_config_obj = config_dict_to_object(summary['best_config'])
            os.makedirs("configs/search_results", exist_ok=True)
            
            from configs.twin_verification_config import save_config
            save_config(best_config_obj, "configs/search_results/best_config.json")
            print(f"   💾 Saved to: configs/search_results/best_config.json")
        
        print(f"\n📋 WORKING CONFIGURATIONS:")
        for i, config_entry in enumerate(self.results.successful_configs):
            config = config_entry['config']
            result = config_entry['result']
            print(f"   {i+1}. {config['name']}: {result['peak_memory_gb']:.1f}GB memory, "
                  f"{result.get('samples_per_second', 0):.1f} samples/s")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Automated Configuration Search')
    parser.add_argument('--gpu_type', choices=['p100', 't4_distributed'], default='p100',
                        help='GPU type to optimize for')
    parser.add_argument('--test_duration', type=int, default=300,
                        help='Duration to test each config (seconds)')
    parser.add_argument('--wandb_entity', type=str, 
                        default='hunchoquavodb-hanoi-university-of-science-and-technology',
                        help='WandB entity for logging')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU.")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"🖥️  Available GPUs: {gpu_count}")
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {gpu_props.name}, {gpu_props.total_memory/1024**3:.1f}GB")
    
    # Generate search space
    if args.gpu_type == 'p100':
        search_space = generate_config_search_space()
        print(f"🎯 Optimizing for Kaggle P100 single GPU")
    else:
        search_space = generate_distributed_t4_search_space()
        print(f"🎯 Optimizing for Kaggle T4 distributed (2 GPUs)")
    
    # Update WandB entity in all configs
    for config in search_space:
        config['WANDB_ENTITY'] = args.wandb_entity
    
    # Run search
    tester = ConfigTester(args.test_duration)
    results = tester.run_search(search_space)
    
    # Save detailed results
    import json
    os.makedirs("configs/search_results", exist_ok=True)
    with open("configs/search_results/detailed_results.json", "w") as f:
        json.dump(results.get_summary(), f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: configs/search_results/detailed_results.json")
    print(f"🎉 Configuration search completed!")


if __name__ == '__main__':
    main() 