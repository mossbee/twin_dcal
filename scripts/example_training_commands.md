# Training Examples for Different Environments

## Local Server (2x RTX 2080Ti) with MLFlow

### Standard Training
```bash
# Start MLFlow server first
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments

# Train with MLFlow tracking
python scripts/train_twin_verification.py \
    --config default \
    --dataset_info data/dataset_infor.json \
    --twin_pairs data/twin_pairs_infor.json
```

### Single GPU Training
```bash
python scripts/train_twin_verification.py \
    --config single_gpu \
    --tracking mlflow
```

### No Tracking (Local only)
```bash
python scripts/train_twin_verification.py \
    --config no_tracking
```

## Kaggle Environment with WandB

### Basic Kaggle Training
```bash
python scripts/train_twin_verification.py \
    --config kaggle \
    --wandb_project "twin-face-verification" \
    --wandb_entity "your-username"
```

### Kaggle with Custom Paths
```bash
python scripts/train_twin_verification.py \
    --config kaggle \
    --dataset_info "/kaggle/input/twin-dataset/dataset_infor.json" \
    --twin_pairs "/kaggle/input/twin-dataset/twin_pairs_infor.json" \
    --wandb_project "twin-verification-experiment-1"
```

## Mixed Environment Training

### Override Tracking Method
```bash
# Use default config but with WandB instead of MLFlow
python scripts/train_twin_verification.py \
    --config default \
    --tracking wandb \
    --wandb_project "local-twin-verification"

# Use Kaggle config but with no tracking
python scripts/train_twin_verification.py \
    --config kaggle \
    --tracking none
```

## Debug and Testing

### Debug Mode (Fast Training)
```bash
python scripts/train_twin_verification.py \
    --config debug \
    --tracking none
```

### Validation Only
```bash
python scripts/train_twin_verification.py \
    --config default \
    --validate_only \
    --resume checkpoints/best_model.pth \
    --tracking none
```

## Distributed Training

### Manual Multi-GPU Launch
```bash
python scripts/train_twin_verification.py \
    --config default \
    --world_size 2 \
    --master_addr localhost \
    --master_port 12355
```

### Using torch.distributed.launch
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=12355 \
    scripts/train_twin_verification.py \
    --config default
```

## Environment-Specific Notes

### Local Server Requirements
- MLFlow server running on localhost:5000
- 2x RTX 2080Ti GPUs available
- Local dataset in `data/` directory

### Kaggle Requirements  
- WandB account and API key
- Dataset uploaded to Kaggle Datasets
- T4 x2 or P100 GPU runtime

### Available Tracking Methods
1. **MLFlow**: `--tracking mlflow` (local privacy-compliant)
2. **WandB**: `--tracking wandb` (cloud-based for Kaggle)
3. **None**: `--tracking none` (TensorBoard only) 