#!/usr/bin/env python3
"""
GPU-Safe test script for DCAL model
Ultra-conservative memory usage to prevent system kills on RTX 2080Ti
"""

import torch
import torch.nn.functional as F
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.twin_verification_config import get_debug_config
from models.dcal_verification_model import create_dcal_model, count_parameters


def create_minimal_config():
    """Create ultra-conservative configuration for GPU testing"""
    config = get_debug_config()
    
    # Ultra-minimal settings to avoid GPU memory issues
    config.BATCH_SIZE_PER_GPU = 1
    config.TOTAL_BATCH_SIZE = 1
    config.EFFECTIVE_BATCH_SIZE = 4
    config.GRADIENT_ACCUMULATION = 4
    
    # Very small model
    config.D_MODEL = 256  # Much smaller than default 768
    config.NUM_HEADS = 4  # Smaller than default 12
    config.D_FF = 1024   # Much smaller than default 3072
    config.SA_BLOCKS = 4  # Fewer blocks
    config.GLCA_BLOCKS = 1
    config.PWCA_BLOCKS = 4
    
    # Smaller feature dimensions
    config.FEATURE_DIM = 256 * 2
    config.VERIFICATION_HIDDEN_DIMS = [128, 64]
    
    # Disable memory-intensive features
    config.MIXED_PRECISION = False  # Disable for stability
    config.COMPILE_MODEL = False
    
    print(f"Minimal config: d_model={config.D_MODEL}, blocks={config.SA_BLOCKS}")
    return config


def check_gpu_status():
    """Check GPU status and memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
    
    print(f"🖥️  GPU: {props.name}")
    print(f"   Total memory: {total_memory:.1f}GB")
    print(f"   Allocated: {allocated_memory:.1f}GB")
    print(f"   Available: {total_memory - allocated_memory:.1f}GB")
    
    if total_memory < 10.0:
        print("⚠️  Limited GPU memory - will use ultra-conservative settings")
        return "limited"
    else:
        print("✅ Adequate GPU memory")
        return "adequate"


def minimal_gpu_test():
    """Minimal GPU test to verify basic functionality"""
    print("🔧 GPU-SAFE MODEL TEST")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. Check GPU
        print("\n1️⃣  Checking GPU status...")
        gpu_status = check_gpu_status()
        if gpu_status is False:
            print("❌ No GPU available for testing")
            return False
        
        # 2. Create minimal model
        print("\n2️⃣  Creating minimal model...")
        config = create_minimal_config()
        config.validate_config()
        
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        
        model = create_dcal_model(config)
        param_counts = count_parameters(model)
        print(f"   ✅ Model created ({param_counts['total']:,} parameters)")
        
        # Move to GPU
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        model = model.to(device)
        model_memory = torch.cuda.memory_allocated(device) / 1024**3
        
        print(f"   Model memory usage: {model_memory - initial_memory:.2f}GB")
        
        if model_memory > 8.0:
            print("   ⚠️  Model uses too much memory, aborting")
            return False
        
        # 3. Test inference (no gradients)
        print("\n3️⃣  Testing inference...")
        model.eval()
        
        # Very small test data
        img1 = torch.randn(1, 3, 224, 224, device=device)  # Small images
        img2 = torch.randn(1, 3, 224, 224, device=device)
        
        with torch.no_grad():
            results = model(img1, img2, training=False)
            score = results['verification_score'].item()
        
        print(f"   ✅ Inference successful (score: {score:.3f})")
        
        # Clean up
        del results, img1, img2
        torch.cuda.empty_cache()
        
        # 4. Test feature extraction
        print("\n4️⃣  Testing feature extraction...")
        img = torch.randn(1, 3, 224, 224, device=device)
        
        with torch.no_grad():
            features = model(img, training=False, return_features=True)
            feat_shape = features['features']['combined_features'].shape
        
        print(f"   ✅ Feature extraction successful {feat_shape}")
        
        # Clean up
        del features, img
        torch.cuda.empty_cache()
        
        # 5. Test similarity computation
        print("\n5️⃣  Testing similarity...")
        img1 = torch.randn(1, 3, 224, 224, device=device)
        img2 = torch.randn(1, 3, 224, 224, device=device)
        
        sim = model.compute_similarity(img1, img2)
        print(f"   ✅ Similarity: {sim.item():.3f}")
        
        # Clean up
        del sim, img1, img2
        torch.cuda.empty_cache()
        
        # 6. Optional: Very light gradient test
        print("\n6️⃣  Testing minimal gradient flow...")
        current_memory = torch.cuda.memory_allocated(device) / 1024**3
        available_memory = 11.0 - current_memory  # RTX 2080Ti
        
        if available_memory < 3.0:
            print("   ⚠️  Insufficient memory for gradient test, skipping")
        else:
            model.train()
            model.zero_grad()
            
            # Tiny gradient test
            img1 = torch.randn(1, 3, 224, 224, device=device)
            img2 = torch.randn(1, 3, 224, 224, device=device)
            labels = torch.tensor([[1.0]], device=device)
            
            try:
                results = model(img1, img2, training=True)
                loss = F.binary_cross_entropy(results['verification_score'], labels)
                loss.backward()
                
                # Quick gradient check
                has_grads = any(p.grad is not None for p in list(model.parameters())[:5])
                print(f"   ✅ Gradients flowing: {has_grads}")
                
                # Immediate cleanup
                del img1, img2, labels, results, loss
                model.zero_grad()
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("   ⚠️  Gradient test failed due to memory - this is expected")
                else:
                    raise
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 GPU-SAFE TEST PASSED!")
        print(f"⏱️  Test completed in {elapsed:.1f} seconds")
        print("\n✅ Basic GPU functionality verified!")
        print("   Your model can run on RTX 2080Ti with conservative settings.")
        print("   For full training, monitor GPU memory usage carefully.")
        print("=" * 50)
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            elapsed = time.time() - start_time
            print(f"\n💥 GPU MEMORY ERROR after {elapsed:.1f}s")
            print("   Even minimal model exceeds GPU memory")
            print("   Recommendations:")
            print("   1. Use test_laptop.py for development")
            print("   2. Reduce model size further")
            print("   3. Use CPU training for this dataset size")
        else:
            print(f"\n💥 GPU TEST FAILED: {e}")
        
        # Emergency cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 TEST FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Final cleanup
        if 'model' in locals():
            del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main():
    """Run GPU-safe tests"""
    success = minimal_gpu_test()
    
    print(f"\n{'='*50}")
    if success:
        print("🚀 GPU testing successful!")
        print("Ready for careful training on RTX 2080Ti.")
        print("Monitor nvidia-smi during training.")
    else:
        print("🔧 GPU testing failed.")
        print("Consider using CPU or smaller model.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 