# Training Examples for Different Environments

**All tracking systems are pre-configured. Just choose your method at runtime.**

## Local Server (2x RTX 2080Ti)

### Standard Training (MLFlow already configured)
```bash
# MLFlow server already deployed - just train
python scripts/train_twin_verification.py --config default

# Or explicitly specify MLFlow
python scripts/train_twin_verification.py \
    --config default \
    --tracking mlflow
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

## Kaggle Environment

### Basic Kaggle Training (WandB already configured)
```bash
# WandB credentials pre-configured
python scripts/train_twin_verification.py --config kaggle
```

### Kaggle with Custom Paths
```bash
python scripts/train_twin_verification.py \
    --config kaggle \
    --dataset_info "/kaggle/input/twin-dataset/dataset_infor.json" \
    --twin_pairs "/kaggle/input/twin-dataset/twin_pairs_infor.json"
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

### Local Server
- ✅ MLFlow server: Already deployed and accessible
- ✅ 2x RTX 2080Ti GPUs: Ready to use
- ✅ Dataset: Located in `data/` directory

### Kaggle Environment  
- ✅ WandB credentials: Pre-configured in config files
- ✅ Dataset: Upload to Kaggle Datasets and use kaggle config
- ✅ GPU Runtime: Works with T4 x2 or P100

### Available Tracking Methods
1. **MLFlow**: `--tracking mlflow` (local privacy-compliant, already deployed)
2. **WandB**: `--tracking wandb` (cloud-based, credentials configured)
3. **None**: `--tracking none` (TensorBoard only, maximum privacy) 