"""
DCAL Utils Package
Utility functions for Twin Face Verification
"""

from .verification_metrics import (
    VerificationMetrics,
    compute_roc_metrics,
    compute_verification_accuracy,
    find_optimal_threshold
)
from .twin_visualization import (
    AttentionVisualizer,
    plot_attention_maps,
    visualize_verification_results
)
from .twin_inference import (
    TwinInferenceEngine,
    load_model_for_inference,
    batch_verification
)
from .distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    reduce_tensor
)

__all__ = [
    'VerificationMetrics',
    'compute_roc_metrics',
    'compute_verification_accuracy', 
    'find_optimal_threshold',
    'AttentionVisualizer',
    'plot_attention_maps',
    'visualize_verification_results',
    'TwinInferenceEngine',
    'load_model_for_inference',
    'batch_verification',
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'reduce_tensor'
] 