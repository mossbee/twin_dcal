# Automated Configuration Search Guide

## Overview

The automated configuration search system finds the optimal balance between model performance and memory usage for your specific GPU setup. It tests configurations from best performance (closest to original paper) to lowest memory usage.

## Quick Start

### Step 1: Run Configuration Search

In your Kaggle notebook:

```bash
# Simple one-line command
!python scripts/run_config_search.py
```

This will:
- 🔍 **Auto-detect your GPU** (P100 or T4)
- 🧪 **Test 8 configurations** (3 minutes each)
- 📊 **Log results to WandB** for analysis
- 💾 **Save best config** to `configs/search_results/best_config.json`

### Step 2: Train with Best Configuration

```bash
# Train with automatically found best configuration
!python scripts/train_twin_verification.py --config search_best --wandb_entity "your-wandb-entity" --wandb_project "dcal-twin-verification"
```

## Advanced Usage

### Custom Search Parameters

```bash
# Longer test duration (5 minutes per config)
!python scripts/automated_config_search.py --gpu_type p100 --test_duration 300 --wandb_entity "your-entity"

# Test distributed T4 configurations
!python scripts/automated_config_search.py --gpu_type t4_distributed --test_duration 180
```

### Load Custom Configuration

```bash
# Train with a specific configuration file
!python scripts/train_twin_verification.py --config_file configs/search_results/best_config.json --wandb_entity "your-entity"
```

## Configuration Search Space

The system tests configurations in this order (best performance → lowest memory):

| Config | Description | Input Size | PWCA Blocks | Expected Memory |
|--------|------------|------------|-------------|-----------------|
| `original_paper` | Original DCAL paper config | 448×448 | 12 | 16GB+ |
| `reduced_pwca_8` | Reduced PWCA blocks to 8 | 448×448 | 8 | 12-14GB |
| `reduced_pwca_6` | Reduced PWCA blocks to 6 | 448×448 | 6 | 10-12GB |
| `conservative_pwca_4` | Conservative PWCA blocks | 448×448 | 4 | 8-10GB |
| `minimal_pwca_2` | Minimal PWCA blocks | 448×448 | 2 | 6-8GB |
| `reduced_dims_512` | Reduced model dimensions | 448×448 | 2 | 4-6GB |
| `small_input_224` | Smaller input size | 224×224 | 2 | 3-4GB |
| `ultra_minimal` | Ultra-minimal baseline | 224×224 | 2 | 2-3GB |

## Understanding Results

### Success Metrics

The system ranks configurations by:
1. **Memory utilization** (40% of score) - prefer higher GPU memory usage
2. **Training speed** (30% of score) - prefer faster samples/second
3. **Model complexity** (30% of score) - prefer larger models if they fit

### WandB Logging

Each configuration test logs:
- Memory usage statistics
- Training speed (samples/second)
- Model architecture details
- Success/failure status
- Error messages for debugging

### Result Files

- `configs/search_results/best_config.json` - Best working configuration
- `configs/search_results/detailed_results.json` - Complete search results

## Troubleshooting

### No Working Configuration Found

If all configurations fail:

1. **Check GPU memory**: Ensure you have enough GPU memory
2. **Reduce test duration**: Try `--test_duration 120` for faster testing
3. **Check WandB setup**: Ensure `WANDB_API_KEY` is in Kaggle secrets
4. **Review error logs**: Check the detailed output for specific errors

### Out of Memory Errors

The system automatically catches OOM errors and continues with smaller configurations. If even `ultra_minimal` fails:

1. **Restart kernel** to clear GPU memory
2. **Check background processes** using other GPU memory
3. **Try smaller batch sizes** in the search space

### Performance Issues

If configurations are too slow:

1. **Reduce test duration**: Use `--test_duration 60` for quick testing
2. **Focus on smaller configs**: The system will find working configs faster
3. **Use distributed training**: Try `--gpu_type t4_distributed` for 2x T4 GPUs

## Custom Configuration Development

### Adding New Configurations

Edit `configs/twin_verification_config.py`:

```python
# Add to generate_config_search_space()
{
    **base_config,
    "name": "custom_config",
    "description": "My custom configuration",
    "INPUT_SIZE": 448,
    "PWCA_BLOCKS": 3,
    "BATCH_SIZE_PER_GPU": 6,
    # ... other parameters
}
```

### Configuration Parameters

Key parameters that affect memory usage:

- `PWCA_BLOCKS`: Most memory-intensive (2-12 blocks)
- `INPUT_SIZE`: Image resolution (224 or 448)
- `BATCH_SIZE_PER_GPU`: Batch size per GPU (1-16)
- `D_MODEL`: Model dimensions (384, 512, 768, 1024)
- `SA_BLOCKS`: Self-attention blocks (6-16)

## Best Practices

1. **Start with automated search**: Don't guess configurations manually
2. **Use WandB logging**: Monitor all experiments for comparison
3. **Test thoroughly**: Let each config run for at least 3 minutes
4. **Keep 448×448 input**: Preserves face detail quality
5. **Monitor memory usage**: Aim for 80-90% GPU memory utilization
6. **Save working configs**: Keep successful configurations for future use

## Example Workflow

```bash
# Step 1: Set up WandB
import os
os.chdir('/kaggle/working/twin_dcal')
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")

# Step 2: Run automated search
!python scripts/run_config_search.py

# Step 3: Review results in WandB dashboard
# Visit: https://wandb.ai/your-entity/dcal-config-search

# Step 4: Train with best configuration
!python scripts/train_twin_verification.py --config search_best --wandb_entity "your-entity" --wandb_project "dcal-twin-verification"
```

This system eliminates the guesswork in configuration tuning and finds the optimal setup for your specific hardware automatically! 🚀 