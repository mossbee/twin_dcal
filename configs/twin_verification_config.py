"""
Twin Face Verification Configuration for DCAL

This configuration file contains all hyperparameters and settings 
optimized for the identical twin face verification task using 2x RTX 2080Ti GPUs.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TwinVerificationConfig:
    """Complete configuration for twin face verification"""
    
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    
    # Dataset paths and structure
    DATASET_INFO: str = "data/dataset_infor.json"
    TWIN_PAIRS_INFO: str = "data/twin_pairs_infor.json"
    
    # External test dataset (mentioned in purpose.md)
    # NOTE: Update these paths to point to your actual external test dataset files
    EXTERNAL_TEST_DATASET: Optional[str] = "data/external_test_dataset_infor.json"  # Path to external test dataset info (same format as main dataset)
    EXTERNAL_TEST_PAIRS: Optional[str] = "data/external_test_twin_pairs.json"    # Path to external test twin pairs (optional, can use main twin pairs)
    USE_EXTERNAL_TEST: bool = True              # Use external test set (as mentioned in purpose.md)
    
    # Dataset statistics (main training dataset only - test dataset is external)
    TOTAL_IDENTITIES: int = 353
    TOTAL_IMAGES: int = 6182
    IMAGES_PER_PERSON_MIN: int = 4
    IMAGES_PER_PERSON_MAX: int = 68
    IMAGES_PER_PERSON_AVG: float = 17.5
    
    # Data splits (maximizing small dataset usage for training only)
    TRAIN_RATIO: float = 0.9    # Use 90% for training (more data for training)
    VAL_RATIO: float = 0.1      # Use 10% for validation
    TEST_RATIO: float = 0.0     # No test split from main dataset - external test set available
    
    # Pair generation strategy
    TWIN_PAIR_RATIO: float = 0.3  # 30% of negatives are twin pairs (hard negatives)
    POSITIVE_NEGATIVE_RATIO: float = 1.0  # 1:1 ratio of positive to negative pairs
    
    # ============================================================================
    # INPUT CONFIGURATION
    # ============================================================================
    
    # Image specifications (already preprocessed to 448x448)
    INPUT_SIZE: int = 448
    INPUT_CHANNELS: int = 3
    PATCH_SIZE: int = 16
    NUM_PATCHES: int = 784  # (448/16)^2 = 28^2 = 784
    SEQUENCE_LENGTH: int = 785  # 784 patches + 1 CLS token
    
    # Image preprocessing
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data augmentation (conservative for faces)
    HORIZONTAL_FLIP_PROB: float = 0.0   # Don't flip faces
    ROTATION_DEGREES: int = 5           # Small rotations only
    COLOR_JITTER_BRIGHTNESS: float = 0.1
    COLOR_JITTER_CONTRAST: float = 0.1
    MIXUP_PROB: float = 0.2             # Light mixup
    CUTMIX_PROB: float = 0.0            # Don't cut faces
    
    # ============================================================================
    # DCAL ARCHITECTURE CONFIGURATION
    # ============================================================================
    
    # Transformer dimensions
    D_MODEL: int = 768
    NUM_HEADS: int = 12
    D_FF: int = 3072
    DROPOUT: float = 0.1
    
    # DCAL specific parameters
    LOCAL_QUERY_RATIO: float = 0.15     # R=15% for facial details
    SA_BLOCKS: int = 12                 # L=12 self-attention blocks
    GLCA_BLOCKS: int = 1                # M=1 global-local cross-attention block
    PWCA_BLOCKS: int = 12               # T=12 pair-wise cross-attention blocks
    
    # Attention rollout
    RESIDUAL_FACTOR: float = 0.5        # For attention rollout computation
    
    # Stochastic depth (regularization)
    STOCHASTIC_DEPTH: float = 0.1       # Increased for small dataset
    
    # ============================================================================
    # HARDWARE & DISTRIBUTED TRAINING CONFIGURATION
    # ============================================================================
    
    # GPU configuration
    WORLD_SIZE: int = 2
    GPUS: List[str] = field(default_factory=lambda: ["cuda:0", "cuda:1"])
    
    # Memory optimization for RTX 2080Ti (11GB each)
    BATCH_SIZE_PER_GPU: int = 8         # Conservative for 448x448 images
    TOTAL_BATCH_SIZE: int = 16          # 8 * 2 GPUs
    GRADIENT_ACCUMULATION: int = 4      # Effective batch size = 64
    EFFECTIVE_BATCH_SIZE: int = 64      # 16 * 4 accumulation steps
    
    # Precision and optimization
    MIXED_PRECISION: bool = True        # Essential for memory efficiency
    COMPILE_MODEL: bool = True          # PyTorch 2.0 optimization
    
    # Distributed training
    DIST_BACKEND: str = 'nccl'
    DIST_URL: str = 'env://'
    MASTER_ADDR: str = 'localhost'
    MASTER_PORT: str = '12355'
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    # Training schedule
    EPOCHS: int = 200                   # More epochs for small dataset
    WARMUP_EPOCHS: int = 20             # Longer warmup for small dataset
    
    # Optimization
    OPTIMIZER: str = "AdamW"
    BASE_LEARNING_RATE: float = 3e-4
    LEARNING_RATE_FORMULA: str = "(3e-4/64) × effective_batch_size"
    WEIGHT_DECAY: float = 0.01
    BETAS: List[float] = field(default_factory=lambda: [0.9, 0.999])
    EPS: float = 1e-8
    
    # Learning rate scheduling
    SCHEDULER: str = "cosine_warmup"
    MIN_LR: float = 1e-6
    COSINE_RESTARTS: bool = False
    
    # Regularization (important for small dataset)
    LABEL_SMOOTHING: float = 0.1
    CLIP_GRAD_NORM: float = 1.0
    
    # ============================================================================
    # LOSS CONFIGURATION
    # ============================================================================
    
    # Loss types and weights
    LOSS_TYPE: str = "combined"
    
    # Individual loss weights
    TRIPLET_MARGIN: float = 0.3
    FOCAL_ALPHA: float = 0.25           # For class imbalance
    FOCAL_GAMMA: float = 2.0
    
    # Multi-task loss weights
    LOSS_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "triplet": 0.4,                 # Triplet loss for embedding learning
        "bce": 0.4,                     # Binary cross-entropy for verification
        "focal": 0.2,                   # Focal loss for hard examples
        "sa_weight": 1.0,               # Self-attention branch weight
        "glca_weight": 1.0,             # Global-local cross-attention weight
        "pwca_weight": 0.3              # Pair-wise cross-attention weight (reduced for small dataset)
    })
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Backbone
    BACKBONE_NAME: str = 'vit_base_patch16_224'
    PRETRAINED: bool = True
    FREEZE_BACKBONE_LAYERS: int = 0     # Don't freeze for face verification
    
    # Verification head
    FEATURE_DIM: int = 768 * 2          # Combined SA + GLCA features
    VERIFICATION_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 256])
    
    # ============================================================================
    # EVALUATION CONFIGURATION
    # ============================================================================
    
    # Evaluation metrics
    EVAL_METRICS: List[str] = field(default_factory=lambda: [
        "verification_accuracy",
        "equal_error_rate", 
        "roc_auc",
        "precision_recall_auc",
        "tar_at_far_001",           # True Accept Rate at 0.1% False Accept Rate
        "tar_at_far_01",            # True Accept Rate at 1% False Accept Rate
        "twin_pair_accuracy"        # Accuracy specifically on twin pairs
    ])
    
    # Threshold optimization
    VERIFICATION_THRESHOLD: float = 0.5  # Will be optimized on validation set
    THRESHOLD_SEARCH_RANGE: List[float] = field(default_factory=lambda: [0.1, 0.9])
    THRESHOLD_SEARCH_STEPS: int = 100
    
    # ============================================================================
    # CHECKPOINTING & LOGGING
    # ============================================================================
    
    # Experiment tracking configuration
    TRACKING_MODE: str = "mlflow"  # Options: "mlflow", "wandb", "none"
    
    # MLFlow configuration (for local privacy-compliant tracking)
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # Local MLFlow server
    MLFLOW_EXPERIMENT_NAME: str = "twin_face_verification"
    
    # WandB configuration (for Kaggle cloud environments)
    WANDB_PROJECT: str = "twin-face-verification"
    WANDB_ENTITY: Optional[str] = 'hunchoquavodb-hanoi-university-of-science-and-technology'  # Your wandb username/team
    WANDB_RUN_NAME: Optional[str] = None  # Auto-generated if None
    WANDB_TAGS: List[str] = field(default_factory=lambda: ["dcal", "face-verification", "twins"])
    
    # General logging
    TENSORBOARD_LOG_DIR: str = "logs/tensorboard"
    LOG_FREQ: int = 100                 # Log every 100 steps
    LOG_GRAD_NORM: bool = True
    LOG_ATTENTION_MAPS: bool = True
    
    # Checkpointing
    SAVE_DIR: str = "checkpoints"
    SAVE_FREQ: int = 10                 # Save every 10 epochs
    SAVE_BEST_ONLY: bool = False
    BEST_METRIC: str = "verification_accuracy"
    EARLY_STOPPING_PATIENCE: int = 30

    # Visualization
    VIS_FREQ: int = 500                 # Visualize attention every 500 steps
    NUM_VIS_SAMPLES: int = 4            # Number of samples to visualize
    
    # ============================================================================
    # INFERENCE CONFIGURATION
    # ============================================================================
    
    # Feature extraction
    EXTRACT_FEATURES_ONLY: bool = False
    FEATURE_EXTRACTION_LAYER: str = "combined"  # "sa", "glca", or "combined"
    
    # Inference optimization
    INFERENCE_BATCH_SIZE: int = 16
    TTA_ENABLED: bool = False           # Test-time augmentation
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================
    
    # DataLoader configuration
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 2
    
    # ============================================================================
    # METHODS
    # ============================================================================
    
    def get_learning_rate(self) -> float:
        """Calculate learning rate based on effective batch size"""
        return (self.BASE_LEARNING_RATE / 64) * self.EFFECTIVE_BATCH_SIZE
    
    def get_output_dir(self, experiment_name: str = "twin_dcal") -> str:
        """Get output directory for experiment"""
        return os.path.join(self.SAVE_DIR, experiment_name)
    
    def get_device_count(self) -> int:
        """Get number of available GPUs"""
        return len(self.GPUS)
    
    def is_distributed(self) -> bool:
        """Check if running in distributed mode"""
        return self.WORLD_SIZE > 1
    
    def validate_config(self) -> None:
        """Validate configuration parameters"""
        # Check batch size constraints
        assert self.TOTAL_BATCH_SIZE == self.BATCH_SIZE_PER_GPU * self.WORLD_SIZE
        assert self.EFFECTIVE_BATCH_SIZE == self.TOTAL_BATCH_SIZE * self.GRADIENT_ACCUMULATION
        
        # Check data split ratios
        if self.USE_EXTERNAL_TEST:
            # When using external test dataset, only train and val ratios need to sum to 1.0
            assert abs(self.TRAIN_RATIO + self.VAL_RATIO - 1.0) < 1e-6, \
                f"Train ratio ({self.TRAIN_RATIO}) + Val ratio ({self.VAL_RATIO}) must sum to 1.0 when using external test"
            assert self.TEST_RATIO == 0.0, \
                "TEST_RATIO should be 0.0 when using external test dataset"
        else:
            # When using internal test split, all ratios must sum to 1.0
            assert abs(self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO - 1.0) < 1e-6, \
                f"Train ({self.TRAIN_RATIO}) + Val ({self.VAL_RATIO}) + Test ({self.TEST_RATIO}) ratios must sum to 1.0"
        
        # Check sequence length
        expected_seq_len = (self.INPUT_SIZE // self.PATCH_SIZE) ** 2 + 1  # +1 for CLS
        assert self.SEQUENCE_LENGTH == expected_seq_len
        
        # Check GPU availability
        assert torch.cuda.is_available(), "CUDA must be available for 2x RTX 2080Ti setup"
        for gpu in self.GPUS:
            gpu_id = int(gpu.split(':')[1])
            assert gpu_id < torch.cuda.device_count(), f"GPU {gpu} not available"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TwinVerificationConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Create default configuration instance
default_config = TwinVerificationConfig()

# Don't validate at module load time - only validate when config is actually used
# This allows single GPU setups to work without the default 2-GPU config failing validation
# default_config.validate_config()  # Commented out to allow single GPU imports


# ============================================================================
# CONFIGURATION VARIANTS
# ============================================================================

def get_debug_config() -> TwinVerificationConfig:
    """Get configuration for debugging with smaller settings"""
    config = TwinVerificationConfig()
    
    # Reduce sizes for debugging
    config.EPOCHS = 5
    config.BATCH_SIZE_PER_GPU = 2
    config.TOTAL_BATCH_SIZE = 4
    config.EFFECTIVE_BATCH_SIZE = 8
    config.GRADIENT_ACCUMULATION = 2
    config.LOG_FREQ = 10
    config.VIS_FREQ = 50
    config.SAVE_FREQ = 2
    
    return config


def get_kaggle_distributed_config() -> TwinVerificationConfig:
    """Configuration for Kaggle distributed training (advanced - requires torch.distributed.launch)"""
    config = TwinVerificationConfig()
    
    # Kaggle distributed setup (for advanced users only)
    config.WORLD_SIZE = 2  # Use both Kaggle GPUs
    config.GPUS = ["cuda:0", "cuda:1"]
    
    # Adjust batch size for distributed training
    config.BATCH_SIZE_PER_GPU = 6  # Good size for T4 GPU memory 
    config.TOTAL_BATCH_SIZE = 12   # 6 * 2 GPUs
    config.GRADIENT_ACCUMULATION = 5  # Effective batch size = 60
    config.EFFECTIVE_BATCH_SIZE = 60
    
    # Disable model compilation for Kaggle compatibility (avoids PyTorch Dynamo issues)
    config.COMPILE_MODEL = False
    
    # WandB tracking for Kaggle
    config.TRACKING_MODE = "wandb"
    config.WANDB_PROJECT = "twin-face-verification-kaggle-distributed"
    config.WANDB_TAGS = ["dcal", "kaggle", "distributed", "twins", "face-verification"]
    
    # Kaggle-specific paths
    config.DATASET_INFO = "/kaggle/input/twin-dataset/dataset_infor.json"
    config.TWIN_PAIRS_INFO = "/kaggle/input/twin-dataset/twin_pairs_infor.json"
    config.SAVE_DIR = "/kaggle/working/checkpoints"
    config.TENSORBOARD_LOG_DIR = "/kaggle/working/logs/tensorboard"
    
    # Reduce epochs for Kaggle time limits
    config.EPOCHS = 100
    config.WARMUP_EPOCHS = 10
    
    return config


def get_single_gpu_config() -> TwinVerificationConfig:
    """Configuration for single GPU training"""
    config = TwinVerificationConfig()
    
    # Single GPU setup
    config.WORLD_SIZE = 1
    config.GPUS = ["cuda:0"]
    
    # Adjust batch sizes
    config.BATCH_SIZE_PER_GPU = 4  # Smaller for single GPU
    config.TOTAL_BATCH_SIZE = 4
    config.GRADIENT_ACCUMULATION = 16  # Higher accumulation to maintain effective batch size
    config.EFFECTIVE_BATCH_SIZE = 64
    
    # Use MLFlow by default
    config.TRACKING_MODE = "mlflow"
    
    return config


def get_kaggle_lite_config() -> TwinVerificationConfig:
    """DCAL-Lite configuration for Kaggle T4 - removes most PWCA blocks to reduce memory"""
    config = TwinVerificationConfig()
    
    # Kaggle single GPU setup
    config.WORLD_SIZE = 1  
    config.GPUS = ["cuda:0"]  
    
    # Conservative batch size
    config.BATCH_SIZE_PER_GPU = 2  # Slightly higher than minimal
    config.TOTAL_BATCH_SIZE = 2    
    config.GRADIENT_ACCUMULATION = 30  # Maintain effective batch size
    config.EFFECTIVE_BATCH_SIZE = 60   # 2 * 30 accumulation steps
    
    # Keep 224x224 for face details
    config.INPUT_SIZE = 224
    config.NUM_PATCHES = 196  # (224/16)^2 = 14^2 = 196
    config.SEQUENCE_LENGTH = 197  # 196 patches + 1 CLS token
    
    # Moderate model dimensions 
    config.D_MODEL = 512      
    config.NUM_HEADS = 8      
    config.D_FF = 2048        
    config.FEATURE_DIM = 512 * 2  
    
    # DCAL-Lite: Drastically reduce PWCA blocks (main memory consumer)
    config.SA_BLOCKS = 8      # Self-attention blocks
    config.PWCA_BLOCKS = 2    # Only 2 PWCA blocks instead of 12 (huge memory saving)
    config.GLCA_BLOCKS = 1    # Keep GLCA for local-global interaction
    
    # Adjust loss weights for fewer PWCA blocks
    config.LOSS_WEIGHTS = {
        "triplet": 0.4,
        "bce": 0.4, 
        "focal": 0.2,
        "sa_weight": 1.0,
        "glca_weight": 1.0,
        "pwca_weight": 0.1  # Reduced weight since fewer PWCA blocks
    }
    
    # Memory optimization
    config.NUM_WORKERS = 2    
    config.PIN_MEMORY = False 
    config.PERSISTENT_WORKERS = False
    config.PREFETCH_FACTOR = 1
    
    # Disable memory-intensive features
    config.COMPILE_MODEL = False
    config.MIXED_PRECISION = True  
    config.LOG_ATTENTION_MAPS = False  
    
    # WandB tracking for Kaggle (API key should be in Kaggle secrets as WANDB_API_KEY)
    config.TRACKING_MODE = "wandb"
    config.WANDB_PROJECT = "dcal-twin-verification"
    config.WANDB_ENTITY = "hunchoquavodb-hanoi-university-of-science-and-technology"
    config.WANDB_TAGS = ["dcal-lite", "kaggle", "twins", "face-verification", "memory-optimized"]
    
    # Kaggle paths
    config.DATASET_INFO = "/kaggle/input/twin-dataset/dataset_infor.json"
    config.TWIN_PAIRS_INFO = "/kaggle/input/twin-dataset/twin_pairs_infor.json"
    config.SAVE_DIR = "/kaggle/working/checkpoints"
    config.TENSORBOARD_LOG_DIR = "/kaggle/working/logs/tensorboard"
    
    # Training schedule
    config.EPOCHS = 100
    config.WARMUP_EPOCHS = 10
    
    return config


def get_no_tracking_config() -> TwinVerificationConfig:
    """Configuration with no external tracking"""
    config = TwinVerificationConfig()
    
    # Disable all tracking
    config.TRACKING_MODE = "none"
    config.MLFLOW_EXPERIMENT_NAME = None
    config.WANDB_PROJECT = None
    
    return config


def get_large_model_config() -> TwinVerificationConfig:
    """Get configuration for larger model (if memory allows)"""
    config = TwinVerificationConfig()
    
    # Larger model settings
    config.D_MODEL = 1024
    config.NUM_HEADS = 16
    config.D_FF = 4096
    config.FEATURE_DIM = 1024 * 2
    
    # Reduce batch size due to memory constraints
    config.BATCH_SIZE_PER_GPU = 4
    config.TOTAL_BATCH_SIZE = 8
    config.EFFECTIVE_BATCH_SIZE = 32
    config.GRADIENT_ACCUMULATION = 4
    
    return config


def get_kaggle_p100_config() -> TwinVerificationConfig:
    """Optimized P100 configuration - balanced performance (8-10GB memory usage)"""
    config = TwinVerificationConfig()
    
    # Kaggle single GPU setup
    config.WORLD_SIZE = 1
    config.GPUS = ["cuda:0"]
    
    # Optimized for 448x448 input with controlled PWCA blocks
    config.INPUT_SIZE = 448  # Keep high resolution for face details
    config.NUM_PATCHES = 784  # (448/16)^2 = 28^2 = 784
    config.SEQUENCE_LENGTH = 785  # 784 patches + 1 CLS token
    
    # Scaled batch size for better memory utilization
    config.BATCH_SIZE_PER_GPU = 8  # 4x larger than working configs
    config.TOTAL_BATCH_SIZE = 8
    config.GRADIENT_ACCUMULATION = 8  # Maintain reasonable effective batch size
    config.EFFECTIVE_BATCH_SIZE = 64  # 8 * 8
    
    # Model architecture - key insight: limit PWCA blocks
    config.D_MODEL = 768  # Standard ViT-Base dimensions
    config.NUM_HEADS = 12
    config.D_FF = 3072
    config.SA_BLOCKS = 8  # Reasonable number of SA blocks
    config.GLCA_BLOCKS = 1  # Keep GLCA for core contribution
    config.PWCA_BLOCKS = 2  # KEY: Limit to 2 blocks like working configs
    
    # Optimized data loading
    config.NUM_WORKERS = 4
    config.PIN_MEMORY = True
    config.PERSISTENT_WORKERS = True
    config.PREFETCH_FACTOR = 2
    
    # Disable model compilation for P100 compatibility
    config.COMPILE_MODEL = False
    config.MIXED_PRECISION = True
    
    # Training settings
    config.EPOCHS = 100
    config.WARMUP_EPOCHS = 10
    config.LR = 2e-4
    config.OPTIMIZER = "AdamW"
    config.SCHEDULER = "cosine_warmup"
    
    # WandB tracking
    config.TRACKING_MODE = "wandb"
    config.WANDB_PROJECT = "dcal-twin-verification"
    config.WANDB_ENTITY = "hunchoquavodb-hanoi-university-of-science-and-technology"
    config.WANDB_TAGS = ["dcal", "kaggle", "twins", "face-verification", "p100-optimized", "448x448"]
    
    # Kaggle paths
    config.DATASET_INFO = "/kaggle/input/twin-dataset/dataset_infor.json"
    config.TWIN_PAIRS_INFO = "/kaggle/input/twin-dataset/twin_pairs_infor.json"
    config.SAVE_DIR = "/kaggle/working/checkpoints"
    config.TENSORBOARD_LOG_DIR = "/kaggle/working/logs/tensorboard"
    
    return config


def get_kaggle_p100_fast_config() -> TwinVerificationConfig:
    """Fast P100 configuration - maximum memory utilization (12-14GB memory usage)"""
    config = get_kaggle_p100_config()
    
    # Larger batch size for maximum memory utilization
    config.BATCH_SIZE_PER_GPU = 12  # 3x larger than base config
    config.TOTAL_BATCH_SIZE = 12
    config.GRADIENT_ACCUMULATION = 6  # Lower accumulation for faster updates
    config.EFFECTIVE_BATCH_SIZE = 72  # 12 * 6
    
    # Enhanced model while keeping PWCA blocks minimal
    config.SA_BLOCKS = 12  # More SA blocks (less memory intensive)
    config.GLCA_BLOCKS = 1  # Keep GLCA
    config.PWCA_BLOCKS = 2  # KEY: Keep only 2 PWCA blocks
    
    # Optimized data loading for speed
    config.NUM_WORKERS = 6
    config.PREFETCH_FACTOR = 4
    
    # Adjusted learning rate for larger effective batch size
    config.LR = 3.37e-4  # Scaled for batch size 72
    
    # Updated tags
    config.WANDB_TAGS = ["dcal", "kaggle", "twins", "face-verification", "p100-fast", "448x448", "memory-optimized"]
    
    return config


def get_kaggle_p100_minimal_config() -> TwinVerificationConfig:
    """Minimal P100 configuration - conservative memory usage (4-6GB memory usage)"""
    config = get_kaggle_p100_config()
    
    # Conservative batch size
    config.BATCH_SIZE_PER_GPU = 4  # Same as ultra-conservative but 448x448
    config.TOTAL_BATCH_SIZE = 4
    config.GRADIENT_ACCUMULATION = 15  # Higher accumulation
    config.EFFECTIVE_BATCH_SIZE = 60  # 4 * 15
    
    # Reduced model complexity
    config.D_MODEL = 512  # Smaller dimensions
    config.NUM_HEADS = 8
    config.D_FF = 2048
    config.SA_BLOCKS = 6  # Fewer SA blocks
    config.GLCA_BLOCKS = 1  # Keep GLCA
    config.PWCA_BLOCKS = 2  # KEY: Keep only 2 PWCA blocks
    
    # Conservative data loading
    config.NUM_WORKERS = 2
    config.PREFETCH_FACTOR = 1
    
    # Adjusted learning rate
    config.LR = 1.5e-4
    
    # Updated tags
    config.WANDB_TAGS = ["dcal", "kaggle", "twins", "face-verification", "p100-minimal", "448x448", "conservative"]
    
    return config


def get_kaggle_p100_max_config() -> TwinVerificationConfig:
    """Maximum P100 configuration - push memory limits (14-15GB memory usage)"""
    config = get_kaggle_p100_config()
    
    # Maximum batch size
    config.BATCH_SIZE_PER_GPU = 16  # Very large batch
    config.TOTAL_BATCH_SIZE = 16
    config.GRADIENT_ACCUMULATION = 4  # Minimal accumulation
    config.EFFECTIVE_BATCH_SIZE = 64  # 16 * 4
    
    # Large model with controlled PWCA
    config.D_MODEL = 1024  # Larger dimensions
    config.NUM_HEADS = 16
    config.D_FF = 4096
    config.SA_BLOCKS = 16  # Many SA blocks
    config.GLCA_BLOCKS = 1  # Keep GLCA
    config.PWCA_BLOCKS = 3  # KEY: Only 3 PWCA blocks (still conservative)
    
    # Maximum data loading
    config.NUM_WORKERS = 8
    config.PREFETCH_FACTOR = 6
    
    # Adjusted learning rate
    config.LR = 2.5e-4
    
    # Updated tags
    config.WANDB_TAGS = ["dcal", "kaggle", "twins", "face-verification", "p100-max", "448x448", "large-model"]
    
    return config


# Remove non-working configs - keep only the optimized ones that work
# get_kaggle_p100_data_optimized_config - REMOVED (didn't work)
# get_kaggle_p100_ultra_conservative_config - REMOVED (replaced by p100_minimal)


def get_local_2080ti_fast_config() -> TwinVerificationConfig:
    """Fast configuration for local 2x RTX 2080Ti setup"""
    config = TwinVerificationConfig()
    
    # Distributed setup
    config.WORLD_SIZE = 2
    config.GPUS = ["cuda:0", "cuda:1"]
    
    # Optimized batch size for speed
    config.BATCH_SIZE_PER_GPU = 6  # Smaller per GPU but distributed
    config.TOTAL_BATCH_SIZE = 12   # 6 * 2 GPUs
    config.GRADIENT_ACCUMULATION = 5  # Lower accumulation
    config.EFFECTIVE_BATCH_SIZE = 60  # 12 * 5
    
    # Reduced model complexity
    config.SA_BLOCKS = 8  # Fewer SA blocks
    config.GLCA_BLOCKS = 1  # Keep GLCA
    config.PWCA_BLOCKS = 8  # Fewer PWCA blocks
    
    # Local MLFlow tracking
    config.TRACKING_MODE = "mlflow"
    config.MLFLOW_TRACKING_URI = "http://localhost:5000"
    config.MLFLOW_EXPERIMENT_NAME = "twin_face_verification_fast"
    
    # Optimized data loading
    config.NUM_WORKERS = 6
    config.PREFETCH_FACTOR = 4
    config.PERSISTENT_WORKERS = True
    config.PIN_MEMORY = True
    
    # Faster training schedule
    config.EPOCHS = 150
    config.WARMUP_EPOCHS = 10
    config.LOG_FREQ = 50
    
    return config


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def print_config_summary(config: TwinVerificationConfig) -> None:
    """Print configuration summary"""
    print("=" * 80)
    print("TWIN FACE VERIFICATION CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {config.TOTAL_IMAGES} images, {config.TOTAL_IDENTITIES} identities")
    print(f"Input Size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f"Architecture: ViT-Base with DCAL (SA:{config.SA_BLOCKS}, GLCA:{config.GLCA_BLOCKS}, PWCA:{config.PWCA_BLOCKS})")
    print(f"Hardware: {config.WORLD_SIZE}x GPU, Batch Size: {config.EFFECTIVE_BATCH_SIZE} (effective)")
    print(f"Training: {config.EPOCHS} epochs, LR: {config.get_learning_rate():.2e}")
    print(f"Mixed Precision: {config.MIXED_PRECISION}")
    print("=" * 80)


def save_config(config: TwinVerificationConfig, save_path: str) -> None:
    """Save configuration to file"""
    import json
    
    config_dict = config.to_dict()
    
    # Convert non-serializable types
    for key, value in config_dict.items():
        if isinstance(value, torch.dtype):
            config_dict[key] = str(value)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(load_path: str) -> TwinVerificationConfig:
    """Load configuration from file"""
    import json
    
    with open(load_path, 'r') as f:
        config_dict = json.load(f)
    
    return TwinVerificationConfig.from_dict(config_dict) 
