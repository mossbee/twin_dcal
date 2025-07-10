# MLFlow Local Setup Guide

## Overview

This guide shows how to set up MLFlow locally for experiment tracking while maintaining **complete privacy** (no external data transmission). MLFlow replaces Weights & Biases (wandb) to comply with the server's "no internet data transmission" policy.

## Quick Start

### 1. Install MLFlow

```bash
pip install mlflow>=2.8.0
```

### 2. Start Local MLFlow Server

```bash
# Start MLFlow tracking server locally
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments

# Or with custom configuration
mlflow server \
    --host localhost \
    --port 5000 \
    --backend-store-uri file:./mlflow_experiments \
    --default-artifact-root file:./mlflow_artifacts \
    --serve-artifacts
```

### 3. Access MLFlow UI

Open your browser and navigate to: `http://localhost:5000`

### 4. Run Training with MLFlow

```bash
# Train with MLFlow tracking
python scripts/train_twin_verification.py \
    --config default \
    --mlflow_uri http://localhost:5000

# Or disable MLFlow completely
python scripts/train_twin_verification.py \
    --config default \
    --mlflow_disabled
```

## Detailed Setup

### Directory Structure

```
project_root/
├── mlflow_experiments/     # Experiment metadata storage
├── mlflow_artifacts/       # Model artifacts and files
├── logs/
│   └── tensorboard/       # TensorBoard logs (local)
└── checkpoints/          # Model checkpoints
```

### MLFlow Configuration

The configuration has been updated to use MLFlow instead of wandb:

```python
# configs/twin_verification_config.py
MLFLOW_TRACKING_URI: str = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME: str = "twin-face-verification"
```

### Advanced MLFlow Setup

#### 1. Persistent Storage with Database

For better performance and reliability, use a database backend:

```bash
# Install database support
pip install psycopg2-binary  # For PostgreSQL
# OR
pip install pymysql         # For MySQL

# Start with PostgreSQL backend
mlflow server \
    --host localhost \
    --port 5000 \
    --backend-store-uri postgresql://username:password@localhost/mlflow \
    --default-artifact-root file:./mlflow_artifacts
```

#### 2. Multiple Experiments

```bash
# View experiments
mlflow experiments list

# Create new experiment
mlflow experiments create --experiment-name "twin-verification-ablation"
```

#### 3. MLFlow with Docker

```dockerfile
# Dockerfile.mlflow
FROM python:3.9-slim

RUN pip install mlflow>=2.8.0 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

```bash
# Run MLFlow in Docker
docker build -f Dockerfile.mlflow -t mlflow-server .
docker run -p 5000:5000 -v $(pwd)/mlflow_data:/mlflow mlflow-server
```

## Privacy & Security Benefits

### ✅ **What MLFlow Does (Locally)**
- **Local storage only**: All data stays on your server
- **No external connections**: No data sent to third parties
- **Complete control**: You own all experiment data
- **Offline capable**: Works without internet connection
- **Data privacy**: Sensitive model information never leaves your system

### ❌ **What wandb Did (Privacy Risk)**
- External data transmission to wandb.ai servers
- Model configurations sent to cloud
- Training metrics uploaded externally  
- Potential IP and sensitive data exposure
- Requires internet connection

## Using MLFlow with the Twin Verification Project

### 1. Training Metrics Logged

```python
# Automatically logged during training:
- Training loss, accuracy, learning rate
- Validation metrics, ROC AUC, EER
- Model parameters and configuration
- Hardware usage and timing
- Gradient norms and optimization stats
```

### 2. Viewing Results

1. **MLFlow UI**: `http://localhost:5000`
   - Compare experiments
   - View metrics plots
   - Download artifacts
   - Compare model parameters

2. **TensorBoard**: `tensorboard --logdir logs/tensorboard`
   - Real-time training curves
   - Network graph visualization
   - Attention map visualization

### 3. Model Management

```python
# Models are automatically logged with:
- Model checkpoints
- Best model artifacts  
- Configuration files
- Training metadata
```

## Troubleshooting

### MLFlow Server Issues

```bash
# Check if server is running
curl http://localhost:5000/health

# View server logs
mlflow server --host localhost --port 5000 --log-level DEBUG

# Clear MLFlow cache
rm -rf ~/.cache/mlflow
```

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different port
   mlflow server --host localhost --port 5001
   
   # Update config
   MLFLOW_TRACKING_URI: str = "http://localhost:5001"
   ```

2. **Permission Errors**
   ```bash
   # Ensure write permissions
   chmod -R 755 mlflow_experiments mlflow_artifacts
   ```

3. **Database Connection Issues**
   ```bash
   # Fallback to file storage
   mlflow server --backend-store-uri file:./mlflow_experiments
   ```

## Migration from wandb

### Code Changes Made

1. **Dependencies**: `wandb` → `mlflow`
2. **Configuration**: `WANDB_PROJECT` → `MLFLOW_EXPERIMENT_NAME`
3. **Initialization**: `wandb.init()` → `mlflow.start_run()`
4. **Logging**: `wandb.log()` → `mlflow.log_metrics()`
5. **Cleanup**: `wandb.finish()` → `mlflow.end_run()`

### Data Migration

If you have existing wandb data:

```python
# Export wandb data (if accessible)
import wandb
api = wandb.Api()
runs = api.runs("your-project")

# Import to MLFlow
import mlflow
for run in runs:
    with mlflow.start_run():
        mlflow.log_params(run.config)
        for key, value in run.summary.items():
            mlflow.log_metric(key, value)
```

## Best Practices

### 1. Experiment Organization
```python
# Use descriptive experiment names
mlflow.set_experiment("twin-verification-baseline")
mlflow.set_experiment("twin-verification-ablation-glca")
mlflow.set_experiment("twin-verification-hyperparameter-search")
```

### 2. Tagging and Metadata
```python
# Tag runs for easy filtering
mlflow.set_tag("model_type", "dcal")
mlflow.set_tag("dataset_size", "6182")
mlflow.set_tag("gpu_type", "rtx_2080ti")
```

### 3. Resource Monitoring
```python
# Log system metrics
mlflow.log_metric("gpu_memory_used", gpu_memory)
mlflow.log_metric("gpu_utilization", gpu_util)
```

## Support

For issues specific to this implementation:
1. Check MLFlow server status: `http://localhost:5000`
2. Review logs in `logs/` directory
3. Verify configuration in `configs/twin_verification_config.py`

For MLFlow general documentation: https://mlflow.org/docs/ 