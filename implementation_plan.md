# DCAL Twin Face Verification - Implementation Plan

## Overview
Adapt Dual Cross-Attention Learning (DCAL) for identical twin face verification using ND TWIN dataset (6,182 images, 353 identities).

## Core Architecture
```
Input (448×448) → ViT Backbone → DCAL Encoder → Verification Head → Binary Output
```

### DCAL Components
1. **Self-Attention (SA)**: L=12 standard transformer blocks
2. **Global-Local Cross-Attention (GLCA)**: M=1 block, R=15% top regions
3. **Pair-Wise Cross-Attention (PWCA)**: T=12 blocks (training only)

## Technical Specifications

### Model Parameters
- **Input**: 448×448×3 images (pre-processed)
- **Patch Size**: 16×16 (784 patches + 1 CLS token)
- **Embedding**: d_model=768, num_heads=12
- **Architecture**: ViT-Base backbone + DCAL encoder + verification head

### Training Configuration
- **Hardware**: 2× RTX 2080Ti (11GB each)
- **Batch Size**: 8 per GPU, effective=64 (with gradient accumulation)
- **Learning Rate**: 3e-4 scaled by batch size
- **Epochs**: 200 (longer for small dataset)
- **Optimizer**: AdamW with cosine scheduling

### Data Strategy
- **Split**: 90% train, 10% val (external test set available)
- **Pairs**: 30% twin negatives (hard), 70% regular negatives
- **Augmentation**: Conservative (no flipping, small rotations)

### Loss Function
Combined loss with weights:
- Triplet Loss: 0.4
- Binary Cross-Entropy: 0.4  
- Focal Loss: 0.2

## Training Environments

### Local Server (2× 2080Ti)
- **Tracking**: MLFlow (http://localhost:5000)
- **Features**: Distributed training, mixed precision
- **Command**: `python scripts/train_twin_verification.py --config default`

### Kaggle Environment
- **Tracking**: WandB (for cloud environments)
- **Hardware**: T4×2 or P100
- **Command**: `python scripts/train_twin_verification.py --config kaggle --wandb_project twin-verification`

### No Tracking Mode
- **Local only**: TensorBoard or no logging
- **Command**: `python scripts/train_twin_verification.py --config default --mlflow_disabled`

## Implementation Status
- ✅ Core model architecture (SA + GLCA + PWCA)
- ✅ Distributed training setup
- ✅ MLFlow local tracking
- ✅ Data loading and preprocessing
- ❌ WandB integration for Kaggle
- ✅ Verification metrics and evaluation

## Key Files
- `src/models/dcal_verification_model.py` - Main model
- `src/modules/attention.py` - GLCA/PWCA implementations  
- `src/training/twin_trainer.py` - Training pipeline
- `configs/twin_verification_config.py` - Configuration
- `scripts/train_twin_verification.py` - Training script

## Quick Start
```bash
# Setup
pip install -r requirements.txt
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments

# Train locally
python scripts/train_twin_verification.py --config default

# Train on Kaggle (needs WandB integration)
python scripts/train_twin_verification.py --config kaggle --wandb_project twin-faces
```

## Evaluation Metrics
- Verification Accuracy
- Equal Error Rate (EER)
- ROC AUC
- TAR @ FAR (0.1%, 1%)
- Twin-specific accuracy