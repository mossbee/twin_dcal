# Twin Face Verification - Configuration Guide

## Overview
This guide helps you configure the three tracking methods: MLFlow, WandB, and No Tracking.

## 🔧 Configuration Settings You Need to Update

### 1. MLFlow Configuration

**Current defaults in `configs/twin_verification_config.py`:**
```python
MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # ⚠️ UPDATE THIS
MLFLOW_EXPERIMENT_NAME: str = "twin_face_verification"  # ✅ Can keep or change
```

**What you need to configure:**
- **MLFlow Server URL**: Update `MLFLOW_TRACKING_URI` to your actual MLFlow server
- **Experiment Name**: Optionally change `MLFLOW_EXPERIMENT_NAME`

**How to configure:**
```bash
# Option 1: Edit the config file directly
# Edit configs/twin_verification_config.py line ~205:
MLFLOW_TRACKING_URI: str = "http://YOUR_MLFLOW_SERVER:PORT"

# Option 2: Override at runtime
python scripts/train_twin_verification.py \
    --config default \
    --mlflow_uri "http://YOUR_MLFLOW_SERVER:PORT"
```

### 2. WandB Configuration

**Current defaults in `configs/twin_verification_config.py`:**
```python
WANDB_PROJECT: str = "twin-face-verification"  # ✅ Can keep or change
WANDB_ENTITY: Optional[str] = None  # ⚠️ UPDATE THIS
WANDB_RUN_NAME: Optional[str] = None  # ✅ Auto-generated
WANDB_TAGS: List[str] = ["dcal", "face-verification", "twins"]  # ✅ Can keep
```

**What you need to configure:**
- **WandB API Key**: Set as environment variable
- **WandB Entity**: Your username or team name
- **Project Name**: Optionally change project name

**How to configure:**

1. **Set WandB API Key** (required):
```bash
# Option 1: Environment variable (recommended)
export WANDB_API_KEY="your_wandb_api_key_here"

# Option 2: Login command
wandb login
```

2. **Update Entity** (recommended):
```bash
# Option 1: Edit config file
# Edit configs/twin_verification_config.py line ~209:
WANDB_ENTITY: str = "your_wandb_username"

# Option 2: Override at runtime
python scripts/train_twin_verification.py \
    --config kaggle \
    --wandb_entity "your_wandb_username" \
    --wandb_project "your_project_name"
```

### 3. No Tracking Configuration
No additional setup required - just use `--tracking none` or `--config no_tracking`.

## 🚀 Testing Your Configuration

### Test MLFlow Connection:
```bash
# Test with a debug run
python scripts/train_twin_verification.py \
    --config debug \
    --tracking mlflow \
    --mlflow_uri "http://YOUR_MLFLOW_SERVER:PORT"
```

### Test WandB Connection:
```bash
# Test WandB login first
wandb login

# Test with a debug run
python scripts/train_twin_verification.py \
    --config debug \
    --tracking wandb \
    --wandb_entity "your_username"
```

### Test No Tracking:
```bash
# Test no external tracking
python scripts/train_twin_verification.py \
    --config debug \
    --tracking none
```

## 📊 Complete Training Examples

### 1. Local Training with MLFlow:
```bash
# Update your MLFlow server URL first
python scripts/train_twin_verification.py \
    --config default \
    --tracking mlflow \
    --mlflow_uri "http://YOUR_MLFLOW_SERVER:PORT"
```

### 2. Kaggle Training with WandB:
```bash
# Set WandB API key first: export WANDB_API_KEY="your_key"
python scripts/train_twin_verification.py \
    --config kaggle \
    --tracking wandb \
    --wandb_entity "your_username" \
    --wandb_project "twin-verification-kaggle"
```

### 3. Privacy-First Training (No Tracking):
```bash
# No external tracking - only TensorBoard
python scripts/train_twin_verification.py \
    --config default \
    --tracking none
```

## 🔍 Configuration Files to Update

### Required Updates:

1. **`configs/twin_verification_config.py`** (lines 205-210):
```python
# MLFlow configuration - UPDATE THIS
MLFLOW_TRACKING_URI: str = "http://YOUR_ACTUAL_MLFLOW_SERVER:PORT"
MLFLOW_EXPERIMENT_NAME: str = "twin_face_verification"  # Optional: change name

# WandB configuration - UPDATE THIS  
WANDB_PROJECT: str = "your-project-name"  # Optional: change project name
WANDB_ENTITY: str = "your_wandb_username"  # REQUIRED for WandB
```

2. **Environment Variables for WandB**:
```bash
# Add to your ~/.bashrc or set before training
export WANDB_API_KEY="your_wandb_api_key_here"
```

## ⚠️ Common Issues and Solutions

### MLFlow Issues:
- **Connection Error**: Check if MLFlow server is running and URL is correct
- **Permission Error**: Ensure your MLFlow server accepts connections from your IP
- **Experiment Not Found**: The experiment will be created automatically

### WandB Issues:
- **Authentication Error**: Run `wandb login` or set `WANDB_API_KEY`
- **Entity Not Found**: Make sure `WANDB_ENTITY` matches your username/team
- **Project Not Found**: Projects are created automatically

### General Issues:
- **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Tracking Disabled**: Check that you're not using `--tracking none` unintentionally

## 🎯 Quick Setup Checklist

- [ ] **MLFlow**: Update `MLFLOW_TRACKING_URI` to your server URL
- [ ] **WandB**: Set `WANDB_API_KEY` environment variable
- [ ] **WandB**: Update `WANDB_ENTITY` to your username
- [ ] **Test**: Run a debug training to verify connections
- [ ] **Ready**: Start full training with your preferred tracking method

## 📚 Additional Configuration Options

All tracking arguments available in the training script:

```bash
python scripts/train_twin_verification.py \
    --config [default|kaggle|single_gpu|no_tracking|debug] \
    --tracking [mlflow|wandb|none] \
    --mlflow_uri "http://server:port" \
    --wandb_project "project-name" \
    --wandb_entity "username" \
    --dataset_info "path/to/train_dataset_infor.json" \
    --twin_pairs "path/to/train_twin_pairs.json" \
    --output_dir "custom/output/path"
``` 