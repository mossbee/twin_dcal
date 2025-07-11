# DCAL Twin Face Verification

## Overview

This project implements **Dual Cross-Attention Learning (DCAL)** for identical twin face verification, adapted from fine-grained visual categorization research. The system determines whether two face images belong to the same person, with particular focus on distinguishing between identical twins.

## 🔒 **Privacy-First & Multi-Environment Design**

- **Local MLFlow tracking**: No external data transmission for privacy-sensitive environments
- **Kaggle WandB support**: Cloud-based tracking for Kaggle competitions and experiments  
- **Flexible tracking**: Three modes - MLFlow, WandB, or none (TensorBoard only)
- **Offline capable**: Works without internet connection in local mode
- **Data sovereignty**: Training data never leaves your server in local mode

## Features

- **DCAL Architecture**: Global-Local Cross-Attention + Pair-Wise Cross-Attention
- **Twin-specific optimization**: Hard negative mining with twin pairs
- **Distributed training**: Multi-GPU support (2x RTX 2080Ti tested)
- **Privacy compliance**: Local MLFlow experiment tracking only
- **Small dataset optimization**: Maximized data utilization for 6,182 images
- **Comprehensive evaluation**: ROC AUC, EER, TAR@FAR metrics

## Architecture

The DCAL model consists of:

1. **Feature Extractor**: Pre-trained backbone (ResNet, Vision Transformer, etc.)
2. **Self-Attention Module**: Captures intra-image dependencies
3. **Global-Local Cross-Attention**: Learns inter-image distinguishing features
4. **Verification Head**: Final classification for twin verification

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training)
- 8GB+ GPU memory recommended

### Quick Setup

```bash
# Clone repository
git clone https://github.com/your-repo/twin-dcal.git
cd twin-dcal

# Install dependencies (privacy-focused - no external tracking)
pip install -r requirements.txt

# Install package
pip install -e .

# Setup local MLFlow server (privacy-compliant)
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments
```

### Detailed Installation

1. **Environment Setup**
   ```bash
   conda create -n twin-dcal python=3.9
   conda activate twin-dcal
   ```

2. **Install Core Dependencies**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

3. **Privacy-Compliant Tracking Setup**
   ```bash
   # Start local MLFlow server (no external data transmission)
   mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments
   
   # Access MLFlow UI at: http://localhost:5000
   ```

4. **Verify Installation**
   ```bash
   python test_model.py
   ```

## Quick Start

### 1. Training

#### Local Server (2x RTX 2080Ti) with MLFlow
```bash
# Start local MLFlow server first
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments

# Multi-GPU training with privacy-compliant tracking
python scripts/train_twin_verification.py \
    --config default \
    --dataset_info data/dataset_infor.json \
    --twin_pairs data/twin_pairs_infor.json

# Single GPU training
python scripts/train_twin_verification.py \
    --config single_gpu \
    --tracking mlflow
```

#### Kaggle Environment with WandB
```bash
# Kaggle training with cloud tracking
python scripts/train_twin_verification.py \
    --config kaggle \
    --wandb_project "twin-face-verification" \
    --wandb_entity "your-username"
```

#### No External Tracking (TensorBoard only)
```bash
# Privacy-first: no external tracking
python scripts/train_twin_verification.py \
    --config no_tracking
```

### 2. Evaluation

```bash
# Evaluate trained model
python scripts/evaluate_verification.py \
    --model checkpoints/best_model.pth \
    --test_data data/test_pairs.json \
    --visualize

# With threshold optimization
python scripts/evaluate_verification.py \
    --model best_model.pth \
    --optimize_threshold \
    --val_data data/val_pairs.json
```

### 3. Feature Extraction

```bash
# Extract features from images
python scripts/extract_features.py \
    --model checkpoints/best_model.pth \
    --images_dir data/faces/ \
    --output features.npz \
    --feature_type combined

# With similarity analysis
python scripts/extract_features.py \
    --model best_model.pth \
    --images_dir data/faces/ \
    --similarity_matrix \
    --visualize_features
```

### 4. Interactive Demo

```bash
# Start interactive demo
python scripts/demo.py --model checkpoints/best_model.pth --interactive

# Single pair verification
python scripts/demo.py \
    --model best_model.pth \
    --img1 face1.jpg \
    --img2 face2.jpg \
    --show_attention
```

## Data Format

### Training Data Structure

```
data/
├── train/
│   ├── person_001/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── person_002/
│   │   └── ...
│   └── ...
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Twin Pairs Information

Create `twin_pairs_infor.json`:

```json
[
    ["person_001", "person_002"],
    ["person_003", "person_004"],
    ...
]
```

### Test Pairs Format

Create test pairs in JSON format:

```json
[
    {
        "img1": "data/test/person_001/img_001.jpg",
        "img2": "data/test/person_002/img_001.jpg",
        "label": 0
    },
    {
        "img1": "data/test/person_001/img_001.jpg",
        "img2": "data/test/person_001/img_002.jpg",
        "label": 1
    }
]
```

## Configuration

### Model Configuration

Example configuration (`configs/dcal_resnet50.json`):

```json
{
    "model": {
        "backbone": "resnet50",
        "pretrained": true,
        "feature_dim": 2048,
        "attention_dim": 512,
        "num_heads": 8,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 100,
        "weight_decay": 0.0001,
        "scheduler": "cosine"
    },
    "data": {
        "image_size": 224,
        "augmentation": true,
        "normalize": true
    }
}
```

## API Usage

### Python API

```python
from twin_dcal import TwinInferenceEngine

# Load trained model
engine = TwinInferenceEngine('checkpoints/best_model.pth')

# Verify a pair of images
result = engine.verify_pair('image1.jpg', 'image2.jpg')
print(f"Similarity: {result['verification_score']:.4f}")
print(f"Match: {result['verification_decision']}")

# Extract features
features = engine.extract_features(['image1.jpg', 'image2.jpg'])
print(f"Feature shape: {features.shape}")

# Batch verification
pairs = [('img1.jpg', 'img2.jpg'), ('img3.jpg', 'img4.jpg')]
results = engine.batch_verify(pairs)
```

### Command Line Interface

```bash
# Use installed console scripts
dcal-train --config config.json --data_dir data/
dcal-evaluate --model model.pth --test_data test.json
dcal-extract --model model.pth --images_dir images/
dcal-demo --model model.pth --interactive
```

## Results

### Performance Metrics

The model is evaluated on several metrics:

- **Verification Accuracy**: Overall accuracy of twin verification
- **Equal Error Rate (EER)**: Point where FAR = FRR
- **ROC AUC**: Area under ROC curve
- **TAR@FAR**: True Accept Rate at specific False Accept Rates

### Visualization

The framework provides comprehensive visualization tools:

- **Attention Maps**: Visualize where the model focuses
- **ROC Curves**: Performance analysis
- **Feature Space**: t-SNE/PCA visualizations
- **Confusion Analysis**: Detailed error analysis

## Distributed Training

### Multi-GPU Training

```bash
# Using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    scripts/train_twin_verification.py \
    --config config.json \
    --distributed

# Using torchrun (PyTorch >= 1.10)
torchrun --nproc_per_node=4 \
    scripts/train_twin_verification.py \
    --config config.json \
    --distributed
```

### Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train_twin_verification.py \
    --config config.json \
    --distributed

# Node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train_twin_verification.py \
    --config config.json \
    --distributed
```

## Experiment Tracking

### Three Tracking Modes

#### 1. MLFlow (Privacy-Compliant Local)
```bash
# Start local MLFlow server
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow_experiments

# Train with MLFlow tracking (local only - no external data)
python scripts/train_twin_verification.py \
    --config default \
    --tracking mlflow
```

#### 2. WandB (Cloud-Based for Kaggle)
```bash
# Train with WandB tracking (for Kaggle environments)
python scripts/train_twin_verification.py \
    --config kaggle \
    --tracking wandb \
    --wandb_project "twin-face-verification"
```

#### 3. No External Tracking (TensorBoard Only)
```bash
# No external tracking - maximum privacy
python scripts/train_twin_verification.py \
    --config default \
    --tracking none
```

### TensorBoard (Always Available)
```bash
# View training logs locally
tensorboard --logdir logs/tensorboard/
```

### Environment-Specific Notes

#### Local Server (Privacy Mode)
🔒 **Complete Privacy**:
- MLFlow server runs locally (no external connections)
- TensorBoard logs stored locally
- No sensitive data transmitted externally
- Complete offline capability

#### Kaggle Environment (Cloud Mode) 
☁️ **Cloud Integration**:
- WandB for experiment tracking and collaboration
- Optimized for Kaggle GPU limits and time constraints
- Easy sharing and comparison of results

## Advanced Features

### Custom Datasets

```python
from twin_dcal import TwinDataModule, TwinVerificationConfig

# Create custom dataset
config = TwinVerificationConfig.from_file('config.json')
data_module = TwinDataModule(config)

# Add custom transforms
data_module.setup_transforms(your_custom_transforms)
```

### Model Customization

```python
from twin_dcal import DCALVerificationModel

# Create model with custom backbone
model = DCALVerificationModel(
    backbone='efficientnet_b4',
    attention_dim=768,
    num_heads=12
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/example/twin-dcal.git
cd twin-dcal

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dcal_twin_verification,
    title={Dual Cross-Attention Learning for Twin Face Verification},
    author={Your Name and Collaborators},
    journal={Journal Name},
    year={2024},
    volume={XX},
    pages={XXX-XXX}
}
```

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Vision Transformer implementations from [timm](https://github.com/rwightman/pytorch-image-models)
- Attention mechanisms inspired by recent transformer architectures

## Contact

For questions and support, please contact:

- **Email**: research@example.com
- **Issues**: [GitHub Issues](https://github.com/example/twin-dcal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/twin-dcal/discussions)

---

**Note**: This is a research implementation. For production use, please ensure proper validation and testing for your specific use case. 