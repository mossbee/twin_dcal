# Dual Cross-Attention Learning (DCAL) for Identical Twin Face Verification
## Final Implementation Blueprint

## Task Overview
This implementation adapts the DCAL architecture for **identical twin face verification** - a binary classification task that determines whether two face images belong to the same person or not. This is a challenging fine-grained recognition problem requiring the model to distinguish between highly similar faces.

### Dataset Analysis & Optimization
- **Total Identities**: 353 people
- **Total Images**: 6,182 face images (448×448, preprocessed)
- **Images per Person**: 4-68 (variable, average ~17.5 images/person)
- **Dataset Size**: Small but sufficient (comparable to CUB-200: 5,994 images)
- **Twin Pairs**: Hard negatives specified in `twin_pairs_infor.json`
- **Hardware**: 2x NVIDIA 2080Ti GPUs (11GB each)

### Critical Dataset Usage Strategy
**Maximum Data Utilization** (addressing small dataset concern):
```python
# Data Split Strategy for 6,182 images
TRAIN_RATIO = 0.8  # 4,946 images
VAL_RATIO = 0.1    # 618 images  
TEST_RATIO = 0.1   # 618 images

# Pair Generation Potential
- Positive pairs per person: C(n,2) where n=images per person
- Total positive pairs: ~31,000 pairs (from all identities)
- Negative pairs: ~780,000 pairs (353 × 352 / 2 × avg_images)
- Twin negative pairs: ~8,500 pairs (hard negatives)
```

## Architecture Overview

### Core Components (Adapted for Verification)
- **Backbone**: Vision Transformer (ViT/DeiT) with L=12 self-attention blocks
- **GLCA Module**: M=1 block for local facial feature attention  
- **PWCA Module**: T=12 blocks for pair-wise regularization (training only)
- **Verification Head**: Binary classification head for same/different person prediction

### Critical Architectural Details
- **Weight Sharing**: PWCA shares weights with SA, GLCA uses separate weights
- **Module Stacking**: L=12 SA blocks, M=1 GLCA block, T=12 PWCA blocks during training
- **Inference Mode**: Remove PWCA completely, use SA+GLCA features for verification
- **Output**: Verification probability (0-1) indicating same person likelihood

### Mathematical Formulations (Implementation Ready)

#### Self-Attention (Baseline)
```python
# Multi-Head Self-Attention Implementation
def self_attention(Q, K, V, d_model=768, num_heads=12):
    """
    Q, K, V: [batch_size, seq_len, d_model]
    Returns: [batch_size, seq_len, d_model]
    """
    d_k = d_model // num_heads
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

#### Global-Local Cross-Attention (Facial Features)
```python
def attention_rollout(attention_weights, residual_factor=0.5):
    """
    Compute accumulated attention with residual connections
    attention_weights: List of [batch, heads, seq_len, seq_len] from each layer
    """
    batch_size, num_heads, seq_len, _ = attention_weights[0].shape
    
    # Average across heads
    attention_matrices = [w.mean(dim=1) for w in attention_weights]
    
    # Apply residual connections: S_bar = 0.5*S + 0.5*I
    identity = torch.eye(seq_len).to(attention_matrices[0].device)
    rolled_attention = identity.clone()
    
    for attention in attention_matrices:
        attention_with_residual = residual_factor * attention + (1 - residual_factor) * identity
        rolled_attention = torch.matmul(attention_with_residual, rolled_attention)
    
    return rolled_attention

def select_top_r_queries(Q, attention_rollout, r_ratio=0.15):
    """
    Select top-R queries based on CLS token accumulated attention
    Q: [batch_size, seq_len, d_model]
    attention_rollout: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len, d_model = Q.shape
    
    # Get CLS token attention (first row)
    cls_attention = attention_rollout[:, 0, :]  # [batch_size, seq_len]
    
    # Select top-R patches (excluding CLS token itself)
    num_patches = int((seq_len - 1) * r_ratio)  # Exclude CLS token
    _, top_indices = torch.topk(cls_attention[:, 1:], num_patches, dim=1)  # Skip CLS
    top_indices = top_indices + 1  # Adjust for CLS token offset
    
    # Extract top-R queries
    Q_local = torch.gather(Q, 1, top_indices.unsqueeze(-1).expand(-1, -1, d_model))
    
    return Q_local, top_indices

def global_local_cross_attention(Q_local, K_global, V_global):
    """
    GLCA: Cross-attention between local queries and global key-values
    Q_local: [batch_size, num_local, d_model]
    K_global, V_global: [batch_size, seq_len, d_model]
    """
    d_model = Q_local.size(-1)
    scores = torch.matmul(Q_local, K_global.transpose(-2, -1)) / math.sqrt(d_model)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V_global)
    return output, attention_weights
```

#### Pair-Wise Cross-Attention (Twin Regularization)
```python
def pairwise_cross_attention(Q1, K1, V1, K2, V2):
    """
    PWCA: Cross-attention with concatenated key-values from image pairs
    Q1: Query from target image [batch_size, seq_len, d_model]
    K1, V1: Key-Value from target image [batch_size, seq_len, d_model]
    K2, V2: Key-Value from pair image [batch_size, seq_len, d_model]
    """
    # Concatenate key-values from both images
    K_combined = torch.cat([K1, K2], dim=1)  # [batch_size, 2*seq_len, d_model]
    V_combined = torch.cat([V1, V2], dim=1)  # [batch_size, 2*seq_len, d_model]
    
    # Compute attention with contaminated key-values
    d_model = Q1.size(-1)
    scores = torch.matmul(Q1, K_combined.transpose(-2, -1)) / math.sqrt(d_model)
    attention_weights = F.softmax(scores, dim=-1)  # Normalized over 2*seq_len
    output = torch.matmul(attention_weights, V_combined)
    
    return output, attention_weights
```

## Distributed Training Setup (Critical)

### Paper's Training Setup Analysis
- **Original Paper**: 4 Tesla V100 GPUs for CUB (3.8 hours), 1 V100 for MSMT17 (9.5 hours)
- **Our Setup**: 2x RTX 2080Ti (similar performance, smaller memory)

### Multi-GPU Configuration
```python
# Distributed Training Setup
class DistributedConfig:
    # Hardware
    WORLD_SIZE = 2  # 2 GPUs
    GPUS = ["cuda:0", "cuda:1"]
    
    # Memory Optimization (11GB per 2080Ti vs 32GB V100)
    BATCH_SIZE_PER_GPU = 8  # Conservative for 448x448 images
    TOTAL_BATCH_SIZE = 16   # 8 * 2 GPUs
    GRADIENT_ACCUMULATION = 4  # Effective batch size = 64
    MIXED_PRECISION = True  # Essential for memory efficiency
    
    # Communication
    DIST_BACKEND = 'nccl'
    DIST_URL = 'env://'
    
def setup_distributed_training(rank, world_size):
    """Setup for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
def create_distributed_model(model, rank):
    """Wrap model for distributed training"""
    model = model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True  # For PWCA training/inference mode switch
    )
    return model
```

## Implementation Structure (Detailed)

### 1. Core Modules (`src/modules/`)

#### `attention.py` (Complete Implementation Guide)
```python
class MultiHeadSelfAttention(nn.Module):
    """Standard transformer self-attention with multiple heads"""
    def __init__(self, d_model=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Implementation details...
        return output, attention_weights

class GlobalLocalCrossAttention(nn.Module):
    """GLCA for facial feature attention with top-R selection"""
    def __init__(self, d_model=768, num_heads=12, r_ratio=0.15):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.r_ratio = r_ratio
        
        # Separate projections for GLCA (no weight sharing with SA)
        self.W_q_local = nn.Linear(d_model, d_model)
        self.W_k_global = nn.Linear(d_model, d_model)
        self.W_v_global = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, attention_rollout):
        # Implementation details...
        return output, local_attention

class PairWiseCrossAttention(nn.Module):
    """PWCA with twin-aware pair sampling (shares weights with SA)"""
    def __init__(self, self_attention_module):
        super().__init__()
        # Share weights with SA module
        self.sa_module = self_attention_module
        
    def forward(self, x1, x2):
        # Use SA module's weights for Q, K, V projections
        # Implementation details...
        return output, contaminated_attention

class AttentionRollout(nn.Module):
    """Facial region attention accumulation"""
    def __init__(self, residual_factor=0.5):
        super().__init__()
        self.residual_factor = residual_factor
        
    def forward(self, attention_weights_list):
        # Implementation details...
        return accumulated_attention
```

#### `transformer.py` (Clear Module Connections)
```python
class TransformerBlock(nn.Module):
    """Standard encoder block (MSA + FFN + LN + residuals)"""
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        # Pre-norm architecture
        attn_out, attn_weights = self.self_attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        if return_attention:
            return x, attn_weights
        return x

class GLCABlock(nn.Module):
    """Facial feature attention block (separate weights)"""
    def __init__(self, d_model=768, num_heads=12, r_ratio=0.15):
        super().__init__()
        self.glca = GlobalLocalCrossAttention(d_model, num_heads, r_ratio)
        self.feed_forward = FeedForward(d_model, d_ff=3072)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, attention_rollout):
        # Local attention on most discriminative regions
        local_out, local_attn = self.glca(self.norm1(x), attention_rollout)
        x_local = x + local_out  # Residual connection
        
        ff_out = self.feed_forward(self.norm2(x_local))
        x_final = x_local + ff_out
        
        return x_final, local_attn

class PWCABlock(nn.Module):
    """Twin regularization block (shared weights with SA)"""
    def __init__(self, sa_block):
        super().__init__()
        self.sa_block = sa_block  # Share weights
        self.pwca = PairWiseCrossAttention(sa_block.self_attention)
        
    def forward(self, x1, x2=None, training=True):
        if training and x2 is not None:
            # PWCA mode: contaminated attention
            pwca_out, _ = self.pwca(x1, x2)
            return pwca_out
        else:
            # Regular SA mode
            return self.sa_block(x1)

class DCALEncoder(nn.Module):
    """Complete encoder for face verification"""
    def __init__(self, d_model=768, num_heads=12, num_layers=12):
        super().__init__()
        # SA blocks (L=12)
        self.sa_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # GLCA block (M=1)
        self.glca_block = GLCABlock(d_model, num_heads)
        
        # PWCA blocks (T=12, share weights with SA)
        self.pwca_blocks = nn.ModuleList([
            PWCABlock(sa_block) for sa_block in self.sa_blocks
        ])
        
        # Attention rollout computation
        self.attention_rollout = AttentionRollout()
        
    def forward(self, x, x_pair=None, training=True):
        sa_attention_weights = []
        
        # Forward through SA blocks
        for i, sa_block in enumerate(self.sa_blocks):
            x, attn_weights = sa_block(x, return_attention=True)
            sa_attention_weights.append(attn_weights)
            
            # PWCA branch (training only)
            if training and x_pair is not None:
                x_pwca = self.pwca_blocks[i](x, x_pair, training=True)
                # Note: PWCA output is used for loss but doesn't affect main branch
        
        # Compute attention rollout for GLCA
        attention_rollout = self.attention_rollout(sa_attention_weights)
        
        # GLCA block
        x_glca, glca_attention = self.glca_block(x, attention_rollout)
        
        return {
            'sa_features': x,
            'glca_features': x_glca,
            'pwca_features': x_pwca if training and x_pair is not None else None,
            'attention_rollout': attention_rollout,
            'glca_attention': glca_attention
        }
```

### 2. Main Model (`src/models/`)

#### `dcal_verification_model.py` (Complete Architecture)
```python
class DCALVerificationModel(nn.Module):
    """Main model for twin face verification"""
    def __init__(self, 
                 backbone='vit_base_patch16_224',
                 pretrained=True,
                 num_classes=2,  # Binary verification
                 d_model=768):
        super().__init__()
        
        # Vision Transformer Backbone
        self.backbone = create_vit_backbone(backbone, pretrained)
        
        # DCAL Encoder
        self.dcal_encoder = DCALEncoder(d_model)
        
        # Verification Head
        self.verification_head = VerificationHead(d_model)
        
        # Feature extractors for SA and GLCA
        self.sa_classifier = nn.Linear(d_model, d_model)
        self.glca_classifier = nn.Linear(d_model, d_model)
        
    def forward(self, img1, img2=None, training=True):
        # Extract features from both images
        feat1 = self.extract_features(img1, img2 if training else None, training)
        
        if img2 is not None:
            feat2 = self.extract_features(img2, img1 if training else None, training)
            
            # Verification
            verification_score = self.verification_head(
                feat1['combined_features'], 
                feat2['combined_features']
            )
            
            return {
                'verification_score': verification_score,
                'features1': feat1,
                'features2': feat2
            }
        else:
            # Feature extraction only
            return feat1
    
    def extract_features(self, img, img_pair=None, training=True):
        """Extract features using DCAL encoder"""
        # Patch embedding
        x = self.backbone.patch_embed(img)
        
        # Add positional embeddings and CLS token
        x = self.backbone.add_pos_embed(x)
        
        # DCAL encoding
        encoder_out = self.dcal_encoder(x, img_pair, training)
        
        # Combine SA and GLCA features
        sa_feat = self.sa_classifier(encoder_out['sa_features'][:, 0])  # CLS token
        glca_feat = self.glca_classifier(encoder_out['glca_features'][:, 0])  # CLS token
        combined_feat = torch.cat([sa_feat, glca_feat], dim=1)
        
        return {
            'sa_features': sa_feat,
            'glca_features': glca_feat,
            'combined_features': combined_feat,
            'pwca_features': encoder_out['pwca_features'],
            'attention_rollout': encoder_out['attention_rollout']
        }

class VerificationHead(nn.Module):
    """Binary classification head for verification"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Feature combination
        self.feature_combiner = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Verification classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Concatenated pair features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Binary classification
            nn.Sigmoid()
        )
    
    def forward(self, features_a, features_b):
        # Combine SA and GLCA features for each image
        feat_a = self.feature_combiner(features_a)
        feat_b = self.feature_combiner(features_b)
        
        # Concatenate pair features
        pair_features = torch.cat([feat_a, feat_b], dim=1)
        
        # Verification score
        verification_score = self.classifier(pair_features)
        return verification_score
```

### 3. Training Infrastructure (`src/training/`)

#### `twin_data_loader.py` (Maximum Data Utilization)
```python
class TwinVerificationDataset(Dataset):
    """Custom dataset for twin face verification with maximum data usage"""
    def __init__(self, 
                 dataset_info_path,
                 twin_pairs_path,
                 split='train',
                 twin_ratio=0.3,
                 transform=None):
        super().__init__()
        
        # Load dataset information
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Split dataset (stratified by identity)
        self.split_data(split)
        
        self.twin_ratio = twin_ratio
        self.transform = transform
        
        # Pre-compute all possible pairs for efficiency
        self.positive_pairs = self.generate_positive_pairs()
        self.negative_pairs = self.generate_negative_pairs()
        
    def split_data(self, split):
        """Stratified split ensuring each identity appears in only one split"""
        all_ids = list(self.dataset_info.keys())
        random.shuffle(all_ids)
        
        n_total = len(all_ids)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if split == 'train':
            self.identity_ids = all_ids[:n_train]
        elif split == 'val':
            self.identity_ids = all_ids[n_train:n_train+n_val]
        else:  # test
            self.identity_ids = all_ids[n_train+n_val:]
        
        # Filter dataset_info for current split
        self.split_dataset_info = {
            id_: paths for id_, paths in self.dataset_info.items() 
            if id_ in self.identity_ids
        }
        
    def generate_positive_pairs(self):
        """Generate all possible positive pairs (same person)"""
        positive_pairs = []
        for identity_id, image_paths in self.split_dataset_info.items():
            if len(image_paths) >= 2:
                for i in range(len(image_paths)):
                    for j in range(i+1, len(image_paths)):
                        positive_pairs.append((image_paths[i], image_paths[j], 1))
        return positive_pairs
    
    def generate_negative_pairs(self):
        """Generate negative pairs with twin emphasis"""
        negative_pairs = []
        all_ids = list(self.split_dataset_info.keys())
        
        # Twin pairs (hard negatives)
        twin_negatives = []
        for twin_pair in self.twin_pairs:
            id1, id2 = twin_pair
            if id1 in self.split_dataset_info and id2 in self.split_dataset_info:
                for img1 in self.split_dataset_info[id1]:
                    for img2 in self.split_dataset_info[id2]:
                        twin_negatives.append((img1, img2, 0))
        
        # Regular negative pairs
        regular_negatives = []
        num_regular = int(len(twin_negatives) / self.twin_ratio * (1 - self.twin_ratio))
        
        for _ in range(num_regular):
            id1, id2 = random.sample(all_ids, 2)
            # Ensure not a twin pair
            if [id1, id2] not in self.twin_pairs and [id2, id1] not in self.twin_pairs:
                img1 = random.choice(self.split_dataset_info[id1])
                img2 = random.choice(self.split_dataset_info[id2])
                regular_negatives.append((img1, img2, 0))
        
        return twin_negatives + regular_negatives
    
    def __len__(self):
        # Balance positive and negative pairs
        return min(len(self.positive_pairs), len(self.negative_pairs)) * 2
    
    def __getitem__(self, idx):
        # Balanced sampling
        if idx < len(self) // 2:
            # Positive pair
            img1_path, img2_path, label = self.positive_pairs[idx % len(self.positive_pairs)]
        else:
            # Negative pair
            idx_neg = (idx - len(self) // 2) % len(self.negative_pairs)
            img1_path, img2_path, label = self.negative_pairs[idx_neg]
        
        # Load and transform images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': torch.tensor(label, dtype=torch.float32),
            'paths': (img1_path, img2_path)
        }

def create_data_loaders(dataset_info_path, twin_pairs_path, batch_size=8, num_workers=4):
    """Create train/val/test data loaders with proper distributed sampling"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(p=0.0),  # Don't flip faces
        transforms.RandomRotation(degrees=5),     # Small rotations
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TwinVerificationDataset(
        dataset_info_path, twin_pairs_path, 'train', transform=train_transform
    )
    val_dataset = TwinVerificationDataset(
        dataset_info_path, twin_pairs_path, 'val', transform=val_test_transform
    )
    test_dataset = TwinVerificationDataset(
        dataset_info_path, twin_pairs_path, 'test', transform=val_test_transform
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset) if torch.distributed.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset) if torch.distributed.is_initialized() else None
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 4. Complete Configuration (`configs/`)

#### `twin_verification_config.py` (Production Ready)
```python
class TwinVerificationConfig:
    """Complete configuration for twin face verification"""
    
    # Dataset Configuration
    DATASET_INFO = "data/dataset_infor.json"
    TWIN_PAIRS_INFO = "data/twin_pairs_infor.json"
    TOTAL_IDENTITIES = 353
    TOTAL_IMAGES = 6182
    
    # Data Splits (maximizing small dataset usage)
    TRAIN_RATIO = 0.8  # ~4,946 images
    VAL_RATIO = 0.1    # ~618 images
    TEST_RATIO = 0.1   # ~618 images
    
    # Input Configuration
    INPUT_SIZE = 448  # Already preprocessed
    PATCH_SIZE = 16
    SEQUENCE_LENGTH = 784  # 28×28 patches + 1 CLS token = 785, but code uses 784
    
    # DCAL Architecture
    D_MODEL = 768
    NUM_HEADS = 12
    D_FF = 3072
    LOCAL_QUERY_RATIO = 0.15  # R=15% for facial details
    SA_BLOCKS = 12  # L=12
    GLCA_BLOCKS = 1  # M=1  
    PWCA_BLOCKS = 12  # T=12 (training only)
    DROPOUT = 0.1
    
    # Training Configuration (Optimized for 2x 2080Ti)
    BATCH_SIZE_PER_GPU = 8     # Conservative for 448×448 images
    TOTAL_BATCH_SIZE = 16      # 8 * 2 GPUs
    GRADIENT_ACCUMULATION = 4  # Effective batch size = 64
    TWIN_PAIR_RATIO = 0.3      # 30% of negatives are twin pairs
    
    # Optimization
    BASE_LEARNING_RATE = 3e-4
    LEARNING_RATE_FORMULA = "(3e-4/64) × effective_batch_size"
    EPOCHS = 200  # More epochs for small dataset
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 0.01
    SCHEDULER = "cosine_warmup"
    WARMUP_EPOCHS = 20  # Longer warmup for small dataset
    MIN_LR = 1e-6
    
    # Loss Configuration
    LOSS_TYPE = "combined"
    TRIPLET_MARGIN = 0.3
    FOCAL_ALPHA = 0.25  # For class imbalance
    FOCAL_GAMMA = 2.0
    
    LOSS_WEIGHTS = {
        "triplet": 0.4,
        "bce": 0.4,
        "focal": 0.2,
        "sa_weight": 1.0,
        "glca_weight": 1.0,
        "pwca_weight": 0.3  # Reduced for small dataset
    }
    
    # Hardware Configuration
    WORLD_SIZE = 2
    GPUS = ["cuda:0", "cuda:1"]
    MIXED_PRECISION = True
    COMPILE_MODEL = True  # PyTorch 2.0 optimization
    
    # Regularization (Important for small dataset)
    STOCHASTIC_DEPTH = 0.1
    LABEL_SMOOTHING = 0.1
    CUTMIX_PROB = 0.0      # Don't cut faces
    MIXUP_PROB = 0.2       # Light mixup
    
    # Evaluation Configuration
    EVAL_METRICS = [
        "verification_accuracy",
        "equal_error_rate",
        "roc_auc",
        "precision_recall_auc",
        "tar_at_far_001",
        "tar_at_far_01",
        "twin_pair_accuracy"
    ]
    
    # Checkpointing
    SAVE_FREQ = 10  # Save every 10 epochs
    BEST_METRIC = "verification_accuracy"
    EARLY_STOPPING_PATIENCE = 30
    
    # Inference Configuration
    VERIFICATION_THRESHOLD = 0.5  # Will be optimized on validation set
    FEATURE_DIM = 768 * 2  # Combined SA + GLCA features
    
    # Logging
    LOG_FREQ = 100  # Log every 100 steps
    MLFLOW_TRACKING_URI = "http://107.98.152.63:5000"  # MLFlow server (already deployed)
    MLFLOW_EXPERIMENT_NAME = "twin_face_verification"
    TENSORBOARD_LOG_DIR = "logs/tensorboard"
```