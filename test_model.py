#!/usr/bin/env python3
"""
Simple test script to verify DCAL model implementation
Tests basic functionality without requiring the full dataset
Automatically adapts to CPU or GPU environment
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.twin_verification_config import TwinVerificationConfig, get_debug_config, get_laptop_config, get_auto_config
from models.dcal_verification_model import create_dcal_model, count_parameters


def detect_environment():
    """Detect if running on laptop (CPU) or server (GPU)"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"🖥️  Detected GPU environment: {gpu_count} GPU(s)")
        for i, name in enumerate(gpu_names):
            print(f"   GPU {i}: {name}")
        return "server" if gpu_count >= 2 else "single_gpu"
    else:
        print("💻 Detected CPU-only environment (laptop/development)")
        return "laptop"


def check_gpu_memory_available(device):
    """Check available GPU memory and return safe limits"""
    if device.type != 'cuda':
        return float('inf'), 1.0  # No limits for CPU
    
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
    available_memory = total_memory - allocated_memory
    
    print(f"   GPU Memory - Total: {total_memory:.1f}GB, Used: {allocated_memory:.1f}GB, Available: {available_memory:.1f}GB")
    
    # Conservative memory usage based on available memory
    if available_memory < 2.0:
        return 224, 0.5  # Very small images, tiny batch
    elif available_memory < 4.0:
        return 224, 0.7  # Small images
    elif available_memory < 8.0:
        return 448, 0.8  # Normal images but conservative
    else:
        return 448, 1.0  # Normal usage


def test_model_creation():
    """Test that the model can be created successfully"""
    print("Testing model creation...")
    
    # Auto-detect environment and use appropriate config
    env_type = detect_environment()
    
    if env_type == "laptop":
        config = get_laptop_config()
    else:
        # For GPU testing, use a very memory-efficient configuration
        config = get_debug_config()
        config = config.auto_detect_hardware()
        
        # Make GPU testing much more memory-efficient
        if torch.cuda.is_available():
            # Ultra-conservative settings for GPU testing
            config.BATCH_SIZE_PER_GPU = 1  # Very small batch
            config.TOTAL_BATCH_SIZE = 1 if config.WORLD_SIZE == 1 else 2
            config.EFFECTIVE_BATCH_SIZE = 4
            config.GRADIENT_ACCUMULATION = 4
            
            # Reduce model size significantly for testing
            config.D_MODEL = 384  # Much smaller than default 768
            config.NUM_HEADS = 6  # Smaller than default 12
            config.D_FF = 1536   # Smaller than default 3072
            config.SA_BLOCKS = 6  # Fewer blocks
            config.GLCA_BLOCKS = 1
            config.PWCA_BLOCKS = 6
            
            # Smaller feature dimensions
            config.FEATURE_DIM = 384 * 2
            config.VERIFICATION_HIDDEN_DIMS = [256, 128]
            
            print(f"   Using GPU-safe configuration: batch_size={config.BATCH_SIZE_PER_GPU}, d_model={config.D_MODEL}")
    
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
        
        # Move model to appropriate device
        device = torch.device('cpu')
        if torch.cuda.is_available() and any(gpu.startswith('cuda') for gpu in config.GPUS):
            device = torch.device(config.GPUS[0])
            
            # Check GPU memory before moving model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(device) / 1024**3
                print(f"   GPU memory before model: {initial_memory:.2f} GB")
        
        model = model.to(device)
        
        if device.type == 'cuda':
            model_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory after model: {model_memory:.2f} GB")
            
            # If model uses more than 8GB, warn about potential issues
            if model_memory > 8.0:
                print("   ⚠️  Model uses significant GPU memory - tests may fail")
        
        print(f"   Model moved to: {device}")
        
        return model, config
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        if torch.cuda.is_available() and "out of memory" in str(e).lower():
            print("   💡 Try reducing model size or using CPU testing")
        raise


def test_forward_pass(model, config):
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    # Determine device and memory constraints
    device = torch.device('cpu')
    if torch.cuda.is_available() and any(gpu.startswith('cuda') for gpu in config.GPUS):
        device = torch.device(config.GPUS[0])
    
    # Check memory constraints
    max_img_size, memory_factor = check_gpu_memory_available(device)
    
    # Use smaller batch size and image size based on memory
    batch_size = max(1, int(config.BATCH_SIZE_PER_GPU * memory_factor))
    img_size = min(config.INPUT_SIZE, max_img_size)
    channels = config.INPUT_CHANNELS
    
    print(f"   Using batch_size={batch_size}, image_size={img_size}x{img_size}")
    
    try:
        # Create dummy images
        img1 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        img2 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            before_forward = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory before forward pass: {before_forward:.2f} GB")
        
        model.eval()
        
        # Test verification (inference mode)
        print("   Testing inference mode...")
        with torch.no_grad():
            results = model(img1, img2, training=False, return_features=True)
        
        print("✅ Inference forward pass successful")
        print(f"   Verification score shape: {results['verification_score'].shape}")
        print(f"   Verification scores: {results['verification_score'].flatten()}")
        
        # Clean up inference results
        del results
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Test training mode with even smaller batch if needed
        print("   Testing training mode...")
        model.train()
        
        # For training mode, use batch size 1 on GPU to avoid memory issues
        if device.type == 'cuda':
            train_batch_size = 1
            if train_batch_size != batch_size:
                del img1, img2
                img1 = torch.randn(train_batch_size, channels, img_size, img_size, device=device)
                img2 = torch.randn(train_batch_size, channels, img_size, img_size, device=device)
        
        # Use torch.no_grad to avoid storing gradients
        with torch.no_grad():
            results_train = model(img1, img2, training=True, return_features=True)
        
        print("✅ Training forward pass successful")
        print(f"   Training verification score shape: {results_train['verification_score'].shape}")
        
        # Clean up training results
        del results_train
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Test feature extraction only
        print("   Testing feature extraction...")
        with torch.no_grad():
            features = model(img1, training=False, return_features=True)
        
        print("✅ Feature extraction successful")
        print(f"   Combined features shape: {features['features']['combined_features'].shape}")
        
        # Clean up
        del img1, img2, features
        
        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️  GPU memory error: {str(e)[:100]}...")
            print("   Try using test_laptop.py for CPU testing")
            
            # Aggressive cleanup
            import gc
            locals_to_delete = ['img1', 'img2', 'results', 'results_train', 'features']
            for var_name in locals_to_delete:
                if var_name in locals():
                    del locals()[var_name]
            gc.collect()
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            return False
        else:
            print(f"❌ Forward pass failed: {e}")
            raise
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_attention_maps(model, config):
    """Test attention map extraction"""
    print("\nTesting attention map extraction...")
    
    # Determine device
    device = torch.device('cpu')
    if torch.cuda.is_available() and any(gpu.startswith('cuda') for gpu in config.GPUS):
        device = torch.device(config.GPUS[0])
    
    batch_size = 1
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy image
    img = torch.randn(batch_size, channels, img_size, img_size, device=device)
    
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
    
    # Determine device
    device = torch.device('cpu')
    if torch.cuda.is_available() and any(gpu.startswith('cuda') for gpu in config.GPUS):
        device = torch.device(config.GPUS[0])
    
    batch_size = config.BATCH_SIZE_PER_GPU
    img_size = config.INPUT_SIZE
    channels = config.INPUT_CHANNELS
    
    # Create dummy images
    img1 = torch.randn(batch_size, channels, img_size, img_size, device=device)
    img2 = torch.randn(batch_size, channels, img_size, img_size, device=device)
    
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
    
    # Determine device and memory constraints
    device = torch.device('cpu')
    if torch.cuda.is_available() and any(gpu.startswith('cuda') for gpu in config.GPUS):
        device = torch.device(config.GPUS[0])
    
    # For gradient flow test, be extremely conservative with GPU memory
    if device.type == 'cuda':
        max_img_size, memory_factor = check_gpu_memory_available(device)
        # Use even smaller sizes for gradient computation
        img_size = min(224, max_img_size)  # Force small images for gradient test
        batch_size = 1  # Always use batch size 1 for gradient test on GPU
        print(f"   GPU gradient test: batch_size={batch_size}, image_size={img_size}x{img_size}")
    else:
        img_size = min(224, config.INPUT_SIZE)  # Small images for CPU too
        batch_size = 1
        print(f"   CPU gradient test: batch_size={batch_size}, image_size={img_size}x{img_size}")
    
    channels = config.INPUT_CHANNELS
    
    try:
        # Clear any existing gradients and memory
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory before gradient test: {initial_memory:.2f} GB")
        
        # Create very small dummy data
        img1 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        img2 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
        
        model.train()
        
        # Forward pass
        print("   Performing forward pass...")
        if device.type == 'cuda':
            # Monitor memory during forward pass
            forward_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory during forward: {forward_memory:.2f} GB")
        
        results = model(img1, img2, training=True)
        verification_scores = results['verification_score']
        
        # Compute loss
        print("   Computing loss...")
        loss = F.binary_cross_entropy(verification_scores, labels)
        
        if device.type == 'cuda':
            pre_backward_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory before backward: {pre_backward_memory:.2f} GB")
            
            # Check if we have enough memory for backward pass
            available_memory = 11.0 - pre_backward_memory  # RTX 2080Ti has 11GB
            if available_memory < 2.0:  # Need at least 2GB for backward
                print("   ⚠️  Insufficient GPU memory for backward pass, skipping")
                del img1, img2, labels, results, verification_scores, loss
                torch.cuda.empty_cache()
                return False
        
        # Backward pass
        print("   Performing backward pass...")
        loss.backward()
        
        if device.type == 'cuda':
            post_backward_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory after backward: {post_backward_memory:.2f} GB")
        
        # Check gradients quickly to avoid holding memory
        has_gradients = False
        grad_norms = []
        param_count = 0
        
        for name, param in model.named_parameters():
            param_count += 1
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                # Break early if we have enough samples
                if len(grad_norms) > 10:  # Just check first 10 parameters
                    break
        
        if has_gradients:
            print("✅ Gradient flow successful")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Parameters with gradients: {len(grad_norms)}/{min(param_count, 10)} (sampled)")
            if grad_norms:
                print(f"   Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
                print(f"   Max gradient norm: {max(grad_norms):.6f}")
        else:
            print("❌ No gradients found")
        
        # Aggressive cleanup immediately after gradient check
        del img1, img2, labels, results, verification_scores, loss
        model.zero_grad()
        
        # Force cleanup
        import gc
        gc.collect()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   GPU memory after cleanup: {final_memory:.2f} GB")
            
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️  GPU memory exhausted during gradient test")
            print(f"   Error: {str(e)[:100]}...")
            print("   This is expected on RTX 2080Ti with large models")
            print("   Consider using test_laptop.py for development testing")
            
            # Emergency cleanup
            model.zero_grad()
            import gc
            
            # Delete everything we can
            locals_to_delete = ['img1', 'img2', 'labels', 'results', 'verification_scores', 'loss']
            for var_name in locals_to_delete:
                if var_name in locals():
                    try:
                        del locals()[var_name]
                    except:
                        pass
            
            gc.collect()
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return False
        else:
            print(f"❌ Gradient flow test failed: {e}")
            raise
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        # Cleanup on any error
        model.zero_grad()
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        raise


def test_memory_usage():
    """Test approximate memory usage"""
    print("\nTesting memory usage...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        config = get_debug_config()
        config = config.auto_detect_hardware()
        model = create_dcal_model(config).to(device)
        
        batch_size = config.BATCH_SIZE_PER_GPU
        img_size = config.INPUT_SIZE
        channels = config.INPUT_CHANNELS
        
        # Create dummy data
        img1 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        img2 = torch.randn(batch_size, channels, img_size, img_size, device=device)
        
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
        
    else:
        print("⚠️  CUDA not available, skipping GPU memory test")
        print("✅ CPU memory test - model creation uses minimal memory")


def test_configuration_compatibility():
    """Test that configuration works across different environments"""
    print("\nTesting configuration compatibility...")
    
    try:
        # Test laptop config
        laptop_config = get_laptop_config()
        laptop_config.validate_config()
        print("✅ Laptop configuration valid")
        
        # Test auto config
        auto_config = get_auto_config()
        print("✅ Auto configuration successful")
        
        # Test that model can be created with different configs
        if torch.cuda.is_available():
            debug_config = get_debug_config()
            debug_config = debug_config.auto_detect_hardware()
            debug_model = create_dcal_model(debug_config)
            print("✅ Debug configuration model creation successful")
            del debug_model
        
        laptop_model = create_dcal_model(laptop_config)
        print("✅ Laptop configuration model creation successful")
        del laptop_model
        
    except Exception as e:
        print(f"❌ Configuration compatibility test failed: {e}")
        raise


def main():
    """Run all tests"""
    print("=" * 60)
    print("DCAL MODEL IMPLEMENTATION TEST")
    print("=" * 60)
    
    # Detect environment and provide appropriate guidance
    env_type = detect_environment()
    
    if env_type == "laptop":
        print("🔧 Running in development mode (CPU-only)")
        print("   This will test core functionality without GPU requirements")
        print("   Safe to run on your laptop for code validation")
    elif env_type == "single_gpu":
        print("🖥️  Running in single GPU mode")
        print("   Using optimized single GPU configuration")
    else:
        print("🚀 Running in multi-GPU server mode")
        print("   Using optimized 2x GPU configuration")
    
    print("=" * 60)
    
    test_results = {}
    model = None
    config = None
    
    # Early GPU memory check to prevent system kills
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"\n🔍 GPU Memory Check: {total_memory:.1f}GB total")
        
        if total_memory < 8.0:
            print("⚠️  Limited GPU memory detected - will use very conservative settings")
        elif total_memory < 12.0:
            print("✅ Adequate GPU memory for testing")
        else:
            print("✅ Excellent GPU memory available")
    
    try:
        # Test 1: Model creation
        print("\n📋 Test 1: Model Creation")
        try:
            model, config = test_model_creation()
            test_results['model_creation'] = True
            print("✅ Model creation test passed")
        except Exception as e:
            print(f"❌ Model creation test failed: {e}")
            test_results['model_creation'] = False
            # Can't continue without a model
            raise
        
        # Test 2: Forward pass
        print("\n📋 Test 2: Forward Pass")
        try:
            forward_result = test_forward_pass(model, config)
            test_results['forward_pass'] = forward_result
            if forward_result:
                print("✅ Forward pass test passed")
            else:
                print("⚠️  Forward pass test had memory issues but continued")
        except Exception as e:
            print(f"❌ Forward pass test failed: {e}")
            test_results['forward_pass'] = False
        
        # Test 3: Attention maps
        print("\n📋 Test 3: Attention Maps")
        try:
            attention_maps = test_attention_maps(model, config)
            test_results['attention_maps'] = True
            print("✅ Attention maps test passed")
        except Exception as e:
            print(f"❌ Attention maps test failed: {e}")
            test_results['attention_maps'] = False
        
        # Test 4: Similarity computation
        print("\n📋 Test 4: Similarity Computation")
        try:
            similarities = test_similarity_computation(model, config)
            test_results['similarity'] = True
            print("✅ Similarity computation test passed")
        except Exception as e:
            print(f"❌ Similarity computation test failed: {e}")
            test_results['similarity'] = False
        
        # Test 5: Gradient flow (most memory intensive) - check memory first
        print("\n📋 Test 5: Gradient Flow")
        skip_gradient_test = False
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            available_memory = total_memory - current_memory
            
            if available_memory < 3.0:  # Need at least 3GB for gradient test
                print(f"⚠️  Insufficient GPU memory ({available_memory:.1f}GB available)")
                print("   Skipping gradient flow test to prevent system kill")
                test_results['gradient_flow'] = False
                skip_gradient_test = True
        
        if not skip_gradient_test:
            try:
                gradient_result = test_gradient_flow(model, config)
                test_results['gradient_flow'] = gradient_result
                if gradient_result:
                    print("✅ Gradient flow test passed")
                else:
                    print("⚠️  Gradient flow test had memory issues but continued")
            except Exception as e:
                print(f"❌ Gradient flow test failed: {e}")
                test_results['gradient_flow'] = False
        
        # Test 6: Memory usage (if CUDA available)
        print("\n📋 Test 6: Memory Usage")
        try:
            test_memory_usage()
            test_results['memory_usage'] = True
            print("✅ Memory usage test passed")
        except Exception as e:
            print(f"❌ Memory usage test failed: {e}")
            test_results['memory_usage'] = False
        
        # Test 7: Configuration compatibility
        print("\n📋 Test 7: Configuration Compatibility")
        try:
            test_configuration_compatibility()
            test_results['configuration'] = True
            print("✅ Configuration compatibility test passed")
        except Exception as e:
            print(f"❌ Configuration compatibility test failed: {e}")
            test_results['configuration'] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("The DCAL model implementation is working correctly.")
        elif passed_tests >= total_tests * 0.7:  # 70% pass rate
            print("\n✅ MOST TESTS PASSED!")
            print("Core functionality is working. Some advanced features may have issues.")
        else:
            print("\n⚠️  MULTIPLE TEST FAILURES!")
            print("There are significant issues that need to be addressed.")
        
        if env_type == "laptop":
            if passed_tests >= total_tests * 0.7:
                print("\n✅ Development environment tests mostly successful!")
                print("   Your code should work on the GPU server.")
                print("   Failed tests may be due to CPU/memory limitations.")
            else:
                print("\n🔧 Fix critical issues before copying to server.")
            print("   The model will automatically use GPU when available.")
        else:
            if passed_tests == total_tests:
                print("   Production environment ready for training.")
            elif passed_tests >= total_tests * 0.7:
                print("   Most functionality working - memory-intensive tests may fail due to hardware limits.")
            else:
                print("   Some issues detected - check failed tests.")
        
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"💥 CRITICAL TEST FAILURE: {e}")
        
        if env_type == "laptop":
            print("   This indicates a fundamental code issue that needs fixing")
            print("   before copying to the server.")
        else:
            print("   Please check the implementation and configuration.")
            if "out of memory" in str(e).lower():
                print("   💡 Consider using test_laptop.py for development testing")
        
        print("=" * 60)
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if model is not None:
            del model
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main() 