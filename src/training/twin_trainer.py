"""
Twin Face Verification Trainer

This module implements the training pipeline for DCAL twin face verification:
1. Distributed training support
2. Mixed precision training  
3. MLFlow experiment tracking (local only - no external data)
4. Comprehensive metrics and checkpointing
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# MLFlow for local experiment tracking only
import mlflow
import mlflow.pytorch

from models.dcal_verification_model import create_dcal_model, count_parameters
from .verification_losses import VerificationLoss
from .twin_data_loader import create_data_loaders


class DistributedTrainer:
    """
    Distributed training manager for multi-GPU training
    Handles setup, synchronization, and cleanup
    """
    
    def __init__(self, config):
        self.config = config
        self.rank = None
        self.world_size = None
        self.local_rank = None
        
    def setup(self, rank: int, world_size: int):
        """Setup distributed training"""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.MASTER_ADDR
        os.environ['MASTER_PORT'] = self.config.MASTER_PORT
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.DIST_BACKEND,
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
        # Print setup info (only on rank 0)
        if self.is_main_process():
            print(f"Distributed training setup complete:")
            print(f"  World size: {world_size}")
            print(f"  Backend: {self.config.DIST_BACKEND}")
            print(f"  Master: {self.config.MASTER_ADDR}:{self.config.MASTER_PORT}")
    
    def cleanup(self):
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
    
    def get_device(self) -> torch.device:
        """Get the device for this process"""
        return torch.device(f'cuda:{self.local_rank}')
    
    def wrap_model(self, model: nn.Module) -> DDP:
        """Wrap model for distributed training"""
        model = model.to(self.get_device())
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True  # For PWCA training/inference mode switch
        )
        return model
    
    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes"""
        if not dist.is_initialized():
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def barrier(self):
        """Synchronize all processes"""
        if dist.is_initialized():
            dist.barrier()


class TwinVerificationTrainer:
    """
    Main trainer class for twin face verification
    Orchestrates the complete training pipeline
    """
    
    def __init__(self, config, distributed_trainer: Optional[DistributedTrainer] = None):
        self.config = config
        self.distributed_trainer = distributed_trainer
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Device setup
        if distributed_trainer:
            self.device = distributed_trainer.get_device()
            self.is_main_process = distributed_trainer.is_main_process()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main_process = True
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Initialize logging
        self.writer = None
        self.mlflow_run = None
        
        if self.is_main_process:
            # TensorBoard
            self.writer = SummaryWriter(log_dir=self.config.TENSORBOARD_LOG_DIR)
            
            # MLFlow (local only)
            if self.config.MLFLOW_EXPERIMENT_NAME:
                try:
                    # Set tracking URI to local server
                    mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
                    
                    # Set experiment
                    mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
                    
                    # Start run
                    self.mlflow_run = mlflow.start_run(
                        run_name=f"dcal_twin_verification_{int(time.time())}"
                    )
                    
                    # Log configuration
                    config_dict = self.config.to_dict()
                    for key, value in config_dict.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(key, value)
                    
                    print(f"MLFlow tracking initialized: {self.config.MLFLOW_TRACKING_URI}")
                except Exception as e:
                    print(f"Warning: MLFlow initialization failed: {e}")
                    print("Training will continue with TensorBoard logging only")
                    self.mlflow_run = None
        
        # Initialize
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        
    def _setup_logging(self):
        """Setup logging infrastructure"""
        if self.is_main_process:
            # TensorBoard
            log_dir = os.path.join(self.config.TENSORBOARD_LOG_DIR, f"run_{int(time.time())}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
            # MLFlow (local server already deployed)
            try:
                mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
                self.mlflow_run = mlflow.start_run(
                    run_name=f"dcal_twin_verification_{int(time.time())}"
                )
                mlflow.log_params(self.config.to_dict())
                print(f"MLFlow run started: {self.mlflow_run.info.run_id}")
            except Exception as e:
                print(f"Warning: Failed to initialize MLFlow: {e}")
                self.mlflow_run = None
    
    def _setup_model(self):
        """Setup model and loss function"""
        # Create model
        self.model = create_dcal_model(self.config)
        
        # Print model info (only on main process)
        if self.is_main_process:
            param_counts = count_parameters(self.model)
            print(f"Model created:")
            print(f"  Total parameters: {param_counts['total']:,}")
            print(f"  Trainable parameters: {param_counts['trainable']:,}")
            
            # Log to MLFlow
            if self.mlflow_run:
                mlflow.log_metrics(param_counts)
        
        # Wrap for distributed training
        if self.distributed_trainer:
            self.model = self.distributed_trainer.wrap_model(self.model)
        else:
            self.model = self.model.to(self.device)
        
        # Compile model (PyTorch 2.0)
        if self.config.COMPILE_MODEL and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Loss function
        self.loss_fn = VerificationLoss(self.config)
        
        # Mixed precision scaler
        if self.config.MIXED_PRECISION:
            self.scaler = GradScaler()
    
    def _setup_data(self):
        """Setup data loaders"""
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        if self.is_main_process:
            print(f"Data loaders created:")
            print(f"  Train batches: {len(self.train_loader)}")
            print(f"  Val batches: {len(self.val_loader)}")
            print(f"  Test batches: {len(self.test_loader)}")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        if self.config.OPTIMIZER == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get_learning_rate(),
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS
            )
        elif self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.get_learning_rate(),
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        # Learning rate scheduler
        if self.config.SCHEDULER == "cosine_warmup":
            # Warmup + Cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.WARMUP_EPOCHS
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.EPOCHS - self.config.WARMUP_EPOCHS,
                eta_min=self.config.MIN_LR
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.WARMUP_EPOCHS]
            )
        else:
            self.scheduler = None
        
        if self.is_main_process:
            print(f"Optimization setup:")
            print(f"  Optimizer: {self.config.OPTIMIZER}")
            print(f"  Learning rate: {self.config.get_learning_rate():.2e}")
            print(f"  Scheduler: {self.config.SCHEDULER}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Initialize metrics
        epoch_metrics = defaultdict(list)
        epoch_start_time = time.time()
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        # Training loop
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}/{self.config.EPOCHS}",
            disable=not self.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Move data to device
            img1 = batch['img1'].to(self.device, non_blocking=True)
            img2 = batch['img2'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.MIXED_PRECISION):
                outputs = self.model(img1, img2, training=True, return_features=True)
                loss, loss_stats = self.loss_fn(outputs, labels, epoch=self.epoch)
            
            # Backward pass
            if self.config.MIXED_PRECISION:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION == 0:
                # Gradient clipping
                if self.config.CLIP_GRAD_NORM > 0:
                    if self.config.MIXED_PRECISION:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.CLIP_GRAD_NORM
                    )
                else:
                    grad_norm = 0.0
                
                # Optimizer step
                if self.config.MIXED_PRECISION:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                
                # Log metrics
                if self.global_step % self.config.LOG_FREQ == 0:
                    self._log_training_step(loss_stats, grad_norm)
            
            # Collect metrics
            for key, value in loss_stats.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            if self.is_main_process:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Batch timing
            batch_time = time.time() - batch_start_time
            epoch_metrics['batch_time'].append(batch_time)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_summary = {
            key: sum(values) / len(values) for key, values in epoch_metrics.items()
        }
        epoch_summary['epoch_time'] = epoch_time
        
        # Learning rate step
        if self.scheduler:
            self.scheduler.step()
        
        return epoch_summary
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_metrics = defaultdict(list)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=not self.is_main_process):
                # Move data to device
                img1 = batch['img1'].to(self.device, non_blocking=True)
                img2 = batch['img2'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.config.MIXED_PRECISION):
                    outputs = self.model(img1, img2, training=False, return_features=True)
                    loss, loss_stats = self.loss_fn(outputs, labels)
                
                # Collect predictions and labels
                predictions = outputs['verification_score'].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
                
                # Collect metrics
                for key, value in loss_stats.items():
                    val_metrics[key].append(value)
        
        # Compute validation metrics
        val_summary = {
            key: sum(values) / len(values) for key, values in val_metrics.items()
        }
        
        # Compute verification metrics
        verification_metrics = self._compute_verification_metrics(all_predictions, all_labels)
        val_summary.update(verification_metrics)
        
        return val_summary
    
    def _compute_verification_metrics(self, predictions: List[float], labels: List[int]) -> Dict[str, float]:
        """Compute verification-specific metrics"""
        import numpy as np
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # ROC AUC
        roc_auc = roc_auc_score(labels, predictions)
        
        # Accuracy with optimal threshold
        optimal_threshold = self._find_optimal_threshold(predictions, labels)
        binary_predictions = (predictions >= optimal_threshold).astype(int)
        accuracy = accuracy_score(labels, binary_predictions)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        
        # Equal Error Rate (EER)
        eer = self._compute_eer(predictions, labels)
        
        # True Accept Rate at specific False Accept Rates
        tar_at_far_001 = self._compute_tar_at_far(predictions, labels, 0.001)
        tar_at_far_01 = self._compute_tar_at_far(predictions, labels, 0.01)
        
        return {
            'roc_auc': roc_auc,
            'verification_accuracy': accuracy,
            'pr_auc': pr_auc,
            'eer': eer,
            'tar_at_far_001': tar_at_far_001,
            'tar_at_far_01': tar_at_far_01,
            'optimal_threshold': optimal_threshold
        }
    
    def _find_optimal_threshold(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal threshold for binary classification"""
        from sklearn.metrics import f1_score
        
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            binary_pred = (predictions >= threshold).astype(int)
            f1 = f1_score(labels, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _compute_eer(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Equal Error Rate"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(labels, predictions)
        fnr = 1 - tpr
        
        # Find point where FPR = FNR
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer
    
    def _compute_tar_at_far(self, predictions: np.ndarray, labels: np.ndarray, target_far: float) -> float:
        """Compute True Accept Rate at given False Accept Rate"""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        
        # Find closest FAR to target
        far_idx = np.argmin(np.abs(fpr - target_far))
        tar = tpr[far_idx]
        
        return tar
    
    def _log_training_step(self, loss_stats: Dict[str, float], grad_norm: float):
        """Log training step metrics"""
        if not self.is_main_process:
            return

        # TensorBoard
        if self.writer:
            for key, value in loss_stats.items():
                self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        # MLFlow (local only)
        if self.mlflow_run:
            log_dict = {f'train_{k}': v for k, v in loss_stats.items()}
            log_dict.update({
                'train_grad_norm': grad_norm,
                'train_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': self.epoch,
                'global_step': self.global_step
            })
            mlflow.log_metrics(log_dict, step=self.global_step)
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        if not self.is_main_process:
            return

        # TensorBoard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'epoch_train/{key}', value, self.epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'epoch_val/{key}', value, self.epoch)

        # MLFlow (local only)
        if self.mlflow_run:
            log_dict = {}
            log_dict.update({f'epoch_train_{k}': v for k, v in train_metrics.items()})
            log_dict.update({f'epoch_val_{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = self.epoch
            mlflow.log_metrics(log_dict, step=self.epoch)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        # Get model state dict (handle DDP wrapping)
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        # Save regular checkpoint
        save_dir = self.config.get_output_dir()
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {self.config.BEST_METRIC}={metrics[self.config.BEST_METRIC]:.4f}")
        
        # Cleanup old checkpoints (keep last 3)
        self._cleanup_checkpoints(save_dir)
    
    def _cleanup_checkpoints(self, save_dir: str, keep_last: int = 3):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        while len(checkpoint_files) > keep_last:
            old_checkpoint = checkpoint_files.pop(0)
            os.remove(os.path.join(save_dir, old_checkpoint))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict (handle DDP wrapping)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"Checkpoint loaded: epoch {self.epoch}, best {self.config.BEST_METRIC}={self.best_metric:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        
        patience_counter = 0
        
        for epoch in range(self.epoch, self.config.EPOCHS):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Check for improvement
            current_metric = val_metrics[self.config.BEST_METRIC]
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.SAVE_FREQ == 0 or is_best:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(all_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Print epoch summary
            if self.is_main_process:
                print(f"Epoch {epoch}: "
                      f"train_loss={train_metrics['total_loss']:.4f}, "
                      f"val_{self.config.BEST_METRIC}={current_metric:.4f}, "
                      f"best={self.best_metric:.4f}")
        
        print(f"Training completed! Best {self.config.BEST_METRIC}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.writer:
            self.writer.close()
        
        if self.mlflow_run:
            mlflow.end_run()
        
        if self.distributed_trainer:
            self.distributed_trainer.cleanup() 