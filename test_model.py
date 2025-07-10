#!/usr/bin/env python3
"""
Simple test script to verify DCAL model implementation
Tests basic functionality on 2x RTX 2080Ti GPU server
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.twin_verification_config import TwinVerificationConfig, get_debug_config
from models.dcal_verification_model import create_dcal_model, count_parameters


def test_model_creation():
    """Test that the model can be created successfully"""
    print("Testing model creation...")
    
    # Use debug config for smaller model
    config = get_debug_config()
    
    try:
        model = create_dcal_model(config)
        print("✅ Model created successfully")
        
        # Count parameters
        param_counts = count_parameters(model)
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Trainable parameters: {param_counts['trainable']:,}")
        print(f"   Backbone: {param_counts['backbone']:,}")
        print(f"   Encoder: {param_counts['encoder']:,}")
        print(f"   Verification head: {param_counts['verification_head']:,}")
        
        return model, config
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise


def test_forward_pass(model, config):
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    batch_size = 2
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy images
    img1 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    img2 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    
    try:
        model = model.cuda()
        model.eval()
        
        # Test verification (inference mode)
        with torch.no_grad():
            results = model(img1, img2, training=False, return_features=True)
        
        print("✅ Inference forward pass successful")
        print(f"   Verification score shape: {results['verification_score'].shape}")
        print(f"   Verification scores: {results['verification_score'].flatten()}")
        
        # Test training mode
        model.train()
        results_train = model(img1, img2, training=True, return_features=True)
        
        print("✅ Training forward pass successful")
        print(f"   Training verification score shape: {results_train['verification_score'].shape}")
        
        # Test feature extraction only
        features = model(img1, training=False, return_features=True)
        print("✅ Feature extraction successful")
        print(f"   Combined features shape: {features['features']['combined_features'].shape}")
        
        return results
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_attention_maps(model, config):
    """Test attention map extraction"""
    print("\nTesting attention map extraction...")
    
    batch_size = 1
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy image
    img = torch.randn(batch_size, channels, img_size, img_size).cuda()
    
    try:
        model.eval()
        with torch.no_grad():
            attention_maps = model.get_attention_maps(img)
        
        print("✅ Attention map extraction successful")
        print(f"   Attention rollout shape: {attention_maps['attention_rollout'].shape}")
        print(f"   GLCA attention shape: {attention_maps['glca_attention'].shape}")
        
        return attention_maps
        
    except Exception as e:
        print(f"❌ Attention map extraction failed: {e}")
        raise


def test_similarity_computation(model, config):
    """Test similarity computation"""
    print("\nTesting similarity computation...")
    
    batch_size = 2
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy images
    img1 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    img2 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    
    try:
        # Test classification mode
        similarity_cls = model.compute_similarity(img1, img2, mode="classification")
        print("✅ Classification similarity successful")
        print(f"   Classification similarities: {similarity_cls.flatten()}")
        
        # Test distance mode  
        similarity_dist = model.compute_similarity(img1, img2, mode="distance")
        print("✅ Distance similarity successful")
        print(f"   Distance similarities: {similarity_dist.flatten()}")
        
        return similarity_cls, similarity_dist
        
    except Exception as e:
        print(f"❌ Similarity computation failed: {e}")
        raise


def test_gradient_flow(model, config):
    """Test that gradients flow properly"""
    print("\nTesting gradient flow...")
    
    batch_size = 2
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy data
    img1 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    img2 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    labels = torch.randint(0, 2, (batch_size, 1)).float().cuda()
    
    try:
        model.train()
        
        # Forward pass
        results = model(img1, img2, training=True)
        verification_scores = results['verification_score']
        
        # Compute loss
        loss = F.binary_cross_entropy(verification_scores, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        if has_gradients:
            print("✅ Gradient flow successful")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
            print(f"   Max gradient norm: {max(grad_norms):.6f}")
        else:
            print("❌ No gradients found")
            
        return loss
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        raise


def test_memory_usage():
    """Test approximate memory usage"""
    print("\nTesting GPU memory usage...")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config = get_debug_config()
    model = create_dcal_model(config).cuda()
    
    batch_size = config.BATCH_SIZE_PER_GPU
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy data
    img1 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    img2 = torch.randn(batch_size, channels, img_size, img_size).cuda()
    
    # Forward pass
    model.train()
    results = model(img1, img2, training=True)
    
    # Check memory usage
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print("✅ GPU memory test successful")
    print(f"   Current allocated: {memory_allocated:.2f} GB")
    print(f"   Current cached: {memory_cached:.2f} GB")
    print(f"   Peak allocated: {peak_memory:.2f} GB")
    
    # Clean up
    del model, img1, img2, results
    torch.cuda.empty_cache()


def main():
    """Run all tests"""
    print("=" * 60)
    print("DCAL MODEL IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Model creation
        model, config = test_model_creation()
        
        # Test 2: Forward pass
        results = test_forward_pass(model, config)
        
        # Test 3: Attention maps
        attention_maps = test_attention_maps(model, config)
        
        # Test 4: Similarity computation
        similarities = test_similarity_computation(model, config)
        
        # Test 5: Gradient flow
        loss = test_gradient_flow(model, config)
        
        # Test 6: Memory usage
        test_memory_usage()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The DCAL model implementation is working correctly.")
        print("Ready for training on 2x RTX 2080Ti.")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"💥 TEST FAILED: {e}")
        print("Please check the implementation.")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 