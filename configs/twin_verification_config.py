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
    TRAIN_DATASET_INFOR: str = "data/train_dataset_infor.json"
    TRAIN_TWIN_PAIRS: str = "data/train_twin_pairs.json"
    
    # External test dataset (mentioned in purpose.md)
    # NOTE: Update these paths to point to your actual external test dataset files
    TEST_DATASET_INFOR: Optional[str] = "data/test_dataset_infor.json"  # Path to external test dataset info (same format as main dataset)
    TEST_TWIN_PAIRS: Optional[str] = "data/test_twin_pairs.json"    # Path to external test twin pairs (optional, can use main twin pairs)
    USE_TEST_SET: bool = True              # Use external test set (as mentioned in purpose.md)
    
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
        if self.USE_TEST_SET:
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
# CONFIGURATION VARIANTS - CLEANED UP
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


def get_no_tracking_config() -> TwinVerificationConfig:
    """Configuration with no external tracking"""
    config = TwinVerificationConfig()
    
    # Disable all tracking
    config.TRACKING_MODE = "none"
    config.MLFLOW_EXPERIMENT_NAME = None
    config.WANDB_PROJECT = None
    
    return config


# ============================================================================
# AUTOMATED CONFIG SEARCH FUNCTIONS
# ============================================================================

def generate_config_search_space() -> List[Dict[str, Any]]:
    """Generate configurations from best performance to lowest memory usage"""
    
    base_config = {
        "WORLD_SIZE": 1,
        "GPUS": ["cuda:0"],
        "TRACKING_MODE": "wandb",
        "WANDB_PROJECT": "dcal-config-search",
        "WANDB_ENTITY": "hunchoquavodb-hanoi-university-of-science-and-technology",
        "COMPILE_MODEL": False,
        "MIXED_PRECISION": True,
        "EPOCHS": 2,  # Short epochs for testing
        "WARMUP_EPOCHS": 0,
        "LOG_FREQ": 10,
        "SAVE_FREQ": 1,
        "NUM_WORKERS": 4,
        "PIN_MEMORY": True,
        "PERSISTENT_WORKERS": True,
        "PREFETCH_FACTOR": 2,
        "TRAIN_DATASET_INFOR": "/kaggle/input/twin-dataset/train_dataset_infor.json",
        "TRAIN_TWIN_PAIRS": "/kaggle/input/twin-dataset/train_twin_pairs.json",
        "SAVE_DIR": "/kaggle/working/checkpoints",
        "TENSORBOARD_LOG_DIR": "/kaggle/working/logs/tensorboard",
    }
    
    # Define search space from best performance to lowest memory
    search_configs = [
        # Config 1: Original paper configuration (highest performance)
        {
            **base_config,
            "name": "original_paper",
            "description": "Original DCAL paper configuration",
            "INPUT_SIZE": 448,
            "D_MODEL": 768,
            "NUM_HEADS": 12,
            "D_FF": 3072,
            "SA_BLOCKS": 12,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 12,
            "BATCH_SIZE_PER_GPU": 16,
            "GRADIENT_ACCUMULATION": 4,
            "WANDB_TAGS": ["config-search", "original-paper", "high-performance"],
        },
        
        # Config 2: Reduced PWCA blocks (main memory saver)
        {
            **base_config,
            "name": "reduced_pwca_8",
            "description": "Reduced PWCA blocks to 8",
            "INPUT_SIZE": 448,
            "D_MODEL": 768,
            "NUM_HEADS": 12,
            "D_FF": 3072,
            "SA_BLOCKS": 12,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 8,
            "BATCH_SIZE_PER_GPU": 12,
            "GRADIENT_ACCUMULATION": 5,
            "WANDB_TAGS": ["config-search", "reduced-pwca", "balanced"],
        },
        
        # Config 3: Further reduced PWCA blocks
        {
            **base_config,
            "name": "reduced_pwca_6",
            "description": "Reduced PWCA blocks to 6",
            "INPUT_SIZE": 448,
            "D_MODEL": 768,
            "NUM_HEADS": 12,
            "D_FF": 3072,
            "SA_BLOCKS": 12,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 6,
            "BATCH_SIZE_PER_GPU": 10,
            "GRADIENT_ACCUMULATION": 6,
            "WANDB_TAGS": ["config-search", "reduced-pwca", "memory-optimized"],
        },
        
        # Config 4: Conservative PWCA blocks (based on working configs)
        {
            **base_config,
            "name": "conservative_pwca_4",
            "description": "Conservative PWCA blocks (4)",
            "INPUT_SIZE": 448,
            "D_MODEL": 768,
            "NUM_HEADS": 12,
            "D_FF": 3072,
            "SA_BLOCKS": 12,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 4,
            "BATCH_SIZE_PER_GPU": 8,
            "GRADIENT_ACCUMULATION": 8,
            "WANDB_TAGS": ["config-search", "conservative-pwca", "stable"],
        },
        
        # Config 5: Minimal PWCA blocks
        {
            **base_config,
            "name": "minimal_pwca_2",
            "description": "Minimal PWCA blocks (2) - known to work",
            "INPUT_SIZE": 448,
            "D_MODEL": 768,
            "NUM_HEADS": 12,
            "D_FF": 3072,
            "SA_BLOCKS": 12,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 8,
            "GRADIENT_ACCUMULATION": 8,
            "WANDB_TAGS": ["config-search", "minimal-pwca", "should-work"],
        },
        
        # Config 6: Reduced model dimensions
        {
            **base_config,
            "name": "reduced_dims_512",
            "description": "Reduced model dimensions to 512",
            "INPUT_SIZE": 448,
            "D_MODEL": 512,
            "NUM_HEADS": 8,
            "D_FF": 2048,
            "SA_BLOCKS": 8,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 12,
            "GRADIENT_ACCUMULATION": 5,
            "WANDB_TAGS": ["config-search", "reduced-dims", "efficient"],
        },
        
        # Config 7: Smaller input size
        {
            **base_config,
            "name": "small_input_224",
            "description": "Smaller input size (224x224)",
            "INPUT_SIZE": 224,
            "D_MODEL": 512,
            "NUM_HEADS": 8,
            "D_FF": 2048,
            "SA_BLOCKS": 8,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 16,
            "GRADIENT_ACCUMULATION": 4,
            "WANDB_TAGS": ["config-search", "small-input", "low-memory"],
        },
        
        # Config 8: Ultra-minimal (guaranteed to work)
        {
            **base_config,
            "name": "ultra_minimal",
            "description": "Ultra-minimal configuration",
            "INPUT_SIZE": 224,
            "D_MODEL": 384,
            "NUM_HEADS": 6,
            "D_FF": 1536,
            "SA_BLOCKS": 6,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 8,
            "GRADIENT_ACCUMULATION": 8,
            "WANDB_TAGS": ["config-search", "ultra-minimal", "baseline"],
        },
    ]
    
    return search_configs


def generate_distributed_t4_search_space() -> List[Dict[str, Any]]:
    """Generate configurations for distributed T4 training"""
    
    base_config = {
        "WORLD_SIZE": 2,
        "GPUS": ["cuda:0", "cuda:1"],
        "TRACKING_MODE": "wandb",
        "WANDB_PROJECT": "dcal-config-search-t4",
        "WANDB_ENTITY": "hunchoquavodb-hanoi-university-of-science-and-technology",
        "COMPILE_MODEL": False,
        "MIXED_PRECISION": True,
        "EPOCHS": 2,
        "WARMUP_EPOCHS": 0,
        "LOG_FREQ": 10,
        "SAVE_FREQ": 1,
        "NUM_WORKERS": 2,
        "PIN_MEMORY": False,
        "PERSISTENT_WORKERS": False,
        "PREFETCH_FACTOR": 1,
        "TRAIN_DATASET_INFOR": "/kaggle/input/twin-dataset/train_dataset_infor.json",
        "TRAIN_TWIN_PAIRS": "/kaggle/input/twin-dataset/train_twin_pairs.json",
        "SAVE_DIR": "/kaggle/working/checkpoints",
        "TENSORBOARD_LOG_DIR": "/kaggle/working/logs/tensorboard",
    }
    
    # Distributed T4 search space
    search_configs = [
        # Config 1: Conservative distributed start
        {
            **base_config,
            "name": "distributed_conservative",
            "description": "Conservative distributed T4 configuration",
            "INPUT_SIZE": 224,
            "D_MODEL": 512,
            "NUM_HEADS": 8,
            "D_FF": 2048,
            "SA_BLOCKS": 8,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 4,
            "GRADIENT_ACCUMULATION": 8,
            "WANDB_TAGS": ["config-search", "distributed-t4", "conservative"],
        },
        
        # Config 2: Scaled up distributed
        {
            **base_config,
            "name": "distributed_scaled",
            "description": "Scaled distributed T4 configuration",
            "INPUT_SIZE": 224,
            "D_MODEL": 512,
            "NUM_HEADS": 8,
            "D_FF": 2048,
            "SA_BLOCKS": 8,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 6,
            "GRADIENT_ACCUMULATION": 5,
            "WANDB_TAGS": ["config-search", "distributed-t4", "scaled"],
        },
        
        # Config 3: Full resolution distributed
        {
            **base_config,
            "name": "distributed_full_res",
            "description": "Full resolution distributed T4",
            "INPUT_SIZE": 448,
            "D_MODEL": 512,
            "NUM_HEADS": 8,
            "D_FF": 2048,
            "SA_BLOCKS": 6,
            "GLCA_BLOCKS": 1,
            "PWCA_BLOCKS": 2,
            "BATCH_SIZE_PER_GPU": 3,
            "GRADIENT_ACCUMULATION": 10,
            "WANDB_TAGS": ["config-search", "distributed-t4", "full-resolution"],
        },
    ]
    
    return search_configs


def config_dict_to_object(config_dict: Dict[str, Any]) -> TwinVerificationConfig:
    """Convert config dictionary to TwinVerificationConfig object"""
    config = TwinVerificationConfig()
    
    # Apply all config values
    for key, value in config_dict.items():
        if key not in ["name", "description"]:  # Skip metadata
            setattr(config, key, value)
    
    # Calculate derived values
    config.TOTAL_BATCH_SIZE = config.BATCH_SIZE_PER_GPU * config.WORLD_SIZE
    config.EFFECTIVE_BATCH_SIZE = config.TOTAL_BATCH_SIZE * config.GRADIENT_ACCUMULATION
    
    # Set sequence length based on input size
    if config.INPUT_SIZE == 224:
        config.NUM_PATCHES = 196  # (224/16)^2
        config.SEQUENCE_LENGTH = 197  # +1 for CLS
    elif config.INPUT_SIZE == 448:
        config.NUM_PATCHES = 784  # (448/16)^2
        config.SEQUENCE_LENGTH = 785  # +1 for CLS
    
    # Set feature dimensions
    config.FEATURE_DIM = config.D_MODEL * 2  # SA + GLCA features
    
    # Set learning rate
    config.LR = (3e-4 / 64) * config.EFFECTIVE_BATCH_SIZE
    
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
