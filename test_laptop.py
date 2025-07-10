#!/usr/bin/env python3
"""
Laptop-optimized test script for DCAL model
Quick validation of core functionality on CPU-only environment
"""

import torch
import torch.nn.functional as F
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.twin_verification_config import get_laptop_config
from models.dcal_verification_model import create_dcal_model, count_parameters


def quick_test():
    """Quick test of core functionality"""
    print("🔧 LAPTOP DEVELOPMENT TEST")
    print("=" * 50)
    
    # Force CPU usage
    if torch.cuda.is_available():
        print("⚠️  CUDA detected but using CPU for laptop testing")
    else:
        print("💻 CPU-only environment detected")
    
    start_time = time.time()
    
    try:
        # 1. Test configuration
        print("\n1️⃣  Testing configuration...")
        config = get_laptop_config()
        config.validate_config()
        print(f"   ✅ Configuration valid (batch size: {config.BATCH_SIZE_PER_GPU})")
        
        # 2. Test model creation
        print("\n2️⃣  Testing model creation...")
        model = create_dcal_model(config)
        param_counts = count_parameters(model)
        print(f"   ✅ Model created ({param_counts['total']:,} parameters)")
        
        # 3. Test basic forward pass
        print("\n3️⃣  Testing forward pass...")
        batch_size = 1  # Very small for quick test
        img1 = torch.randn(batch_size, 3, 448, 448)
        img2 = torch.randn(batch_size, 3, 448, 448)
        
        model.eval()
        with torch.no_grad():
            # Test inference mode
            results = model(img1, img2, training=False)
            score = results['verification_score'].item()
            print(f"   ✅ Inference successful (score: {score:.3f})")
            
            # Test feature extraction
            features = model(img1, training=False, return_features=True)
            feat_shape = features['features']['combined_features'].shape
            print(f"   ✅ Feature extraction successful ({feat_shape})")
        
        # 4. Test training mode
        print("\n4️⃣  Testing training mode...")
        model.train()
        
        # Single training step
        results = model(img1, img2, training=True)
        labels = torch.tensor([[1.0]])  # Dummy label
        loss = F.binary_cross_entropy(results['verification_score'], labels)
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"   ✅ Training mode successful (loss: {loss.item():.4f}, gradients: {has_grads})")
        
        # 5. Test similarity computation
        print("\n5️⃣  Testing similarity computation...")
        sim_cls = model.compute_similarity(img1, img2, mode="classification")
        sim_dist = model.compute_similarity(img1, img2, mode="distance")
        print(f"   ✅ Similarity computation successful")
        print(f"      Classification: {sim_cls.item():.3f}")
        print(f"      Distance: {sim_dist.item():.3f}")
        
        # 6. Test imports and dependencies
        print("\n6️⃣  Testing imports...")
        from modules.attention import MultiHeadSelfAttention, GlobalLocalCrossAttention
        from modules.transformer import DCALEncoder, TransformerBlock
        from modules.backbone import VisionTransformerBackbone
        print("   ✅ All module imports successful")
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 ALL LAPTOP TESTS PASSED!")
        print(f"⏱️  Test completed in {elapsed:.1f} seconds")
        print("\n✅ Your code is ready for the GPU server!")
        print("   The model will automatically use GPUs when available.")
        print("   Copy your code to the server and run the full training.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 LAPTOP TEST FAILED after {elapsed:.1f}s: {e}")
        print("\n🔧 Fix the issue before copying to server.")
        print("   This error would also occur on the GPU server.")
        import traceback
        traceback.print_exc()
        return False


def test_config_variations():
    """Test different configuration scenarios"""
    print("\n🔬 Testing configuration variations...")
    
    try:
        # Test with different batch sizes
        config = get_laptop_config()
        
        # Test very small batch
        config.BATCH_SIZE_PER_GPU = 1
        config.TOTAL_BATCH_SIZE = 1
        model = create_dcal_model(config)
        print("   ✅ Batch size 1 configuration works")
        del model
        
        # Test slightly larger batch
        config.BATCH_SIZE_PER_GPU = 2
        config.TOTAL_BATCH_SIZE = 2
        model = create_dcal_model(config)
        print("   ✅ Batch size 2 configuration works")
        del model
        
        print("   ✅ Configuration variations successful")
        
    except Exception as e:
        print(f"   ❌ Configuration variation failed: {e}")
        raise


def memory_check():
    """Simple memory usage check"""
    print("\n💾 Memory usage check...")
    
    import psutil
    import gc
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create and delete model
    config = get_laptop_config()
    model = create_dcal_model(config)
    
    # Test forward pass
    img = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        _ = model(img, training=False, return_features=True)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Clean up
    del model, img
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"   Initial memory: {initial_memory:.1f} MB")
    print(f"   Peak memory: {peak_memory:.1f} MB")
    print(f"   Final memory: {final_memory:.1f} MB")
    print(f"   Memory increase: {peak_memory - initial_memory:.1f} MB")
    
    if peak_memory - initial_memory < 2000:  # Less than 2GB
        print("   ✅ Memory usage reasonable for laptop")
    else:
        print("   ⚠️  High memory usage - may be slow on laptop")


def main():
    """Run laptop tests"""
    success = quick_test()
    
    if success:
        try:
            test_config_variations()
            memory_check()
        except Exception as e:
            print(f"\nAdvanced tests failed: {e}")
            print("Core functionality works, but check advanced features.")
    
    print(f"\n{'='*50}")
    if success:
        print("🚀 Ready to deploy to GPU server!")
    else:
        print("🔧 Fix issues before server deployment.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 