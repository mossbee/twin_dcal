# DCAL Twin Face Verification - Development Guide

## Development Workflow: Laptop → Server

This guide explains how to develop on your **CPU-only laptop** and deploy to your **2x RTX 2080Ti GPU server** with no internet access.

## 🔄 Development Workflow

### 1. **Laptop Development** (CPU-only)
- Write and test code on your laptop
- Use CPU-optimized configurations  
- Validate core functionality without GPU requirements
- No need for actual dataset during development

### 2. **Server Deployment** 
- Copy code to GPU server
- Code automatically detects GPU hardware
- Uses optimized GPU configurations
- Ready for full training with actual dataset

---

## 🧪 Testing Scripts

### **Quick Laptop Test** (Recommended for development)
```bash
python test_laptop.py
```
**Purpose**: Fast validation of core functionality on CPU
- ⏱️ Runs in ~10-30 seconds
- 💾 Uses minimal memory
- ✅ Tests all critical components
- 🔧 Perfect for iterative development

### **GPU-Safe Test** (Recommended for RTX 2080Ti server)
```bash
python test_gpu_safe.py
```
**Purpose**: Conservative GPU testing to prevent system kills
- 🛡️ Ultra-conservative memory usage
- 📊 Minimal model size to fit in 11GB VRAM
- 🚀 Verifies GPU functionality without crashes
- ⚠️ Use this first on RTX 2080Ti to check compatibility

### **Comprehensive Test** (Both laptop and server)
```bash
python test_model.py
```
**Purpose**: Full testing that adapts to available hardware
- 🔍 Auto-detects CPU vs GPU environment
- 📊 Comprehensive functionality testing
- 💻 Safe to run on laptop (uses CPU)
- 🖥️ Uses GPU when available (may use lots of memory)

---

## ⚙️ Configurations

### **Automatic Configuration** (Recommended)
```python
from configs.twin_verification_config import get_auto_config
config = get_auto_config()  # Automatically detects hardware
```

### **Laptop-Specific Configuration**
```python
from configs.twin_verification_config import get_laptop_config
config = get_laptop_config()  # Optimized for CPU development
```
- Batch size: 2 (very small for CPU)
- Model size: Reduced for faster testing
- Mixed precision: Disabled
- GPU features: Disabled

### **Server Configuration**
```python
from configs.twin_verification_config import get_server_config
config = get_server_config()  # Optimized for 2x RTX 2080Ti
```
- Batch size: 8 per GPU (16 total)
- Full model size: Production ready
- Mixed precision: Enabled
- Multi-GPU: Enabled

---

## 🚀 Quick Start

### **On Your Laptop** (Development)

1. **Test core functionality**:
   ```bash
   python test_laptop.py
   ```

2. **If tests pass**: Your code is ready! ✅

3. **If tests fail**: Fix the issues before copying to server

4. **Test configurations**:
   ```bash
   python -c "
   from configs.twin_verification_config import get_laptop_config
   config = get_laptop_config()
   config.validate_config()
   print('✅ Configuration valid')
   "
   ```

### **On Your Server** (Production)

1. **Copy your code** to the server

2. **Test GPU compatibility first** (important for RTX 2080Ti):
   ```bash
   python test_gpu_safe.py  # Use this FIRST
   ```
   Should show: "🎉 GPU-SAFE TEST PASSED!"

3. **If GPU-safe test passes**, try comprehensive test:
   ```bash
   python test_model.py
   ```
   Should detect: "🖥️ Detected GPU environment: 2 GPU(s)"

4. **Start training**:
   ```bash
   python train.py  # Will automatically use GPU configuration
   ```

---

## 🔧 Development Tips

### **Iterative Development on Laptop**
1. Make code changes
2. Run `python test_laptop.py` (10 seconds)
3. Fix any issues
4. Repeat until tests pass
5. Copy to server when ready

### **RTX 2080Ti Memory Management** ⚠️
- **Important**: RTX 2080Ti has only 11GB VRAM
- Always run `test_gpu_safe.py` first on the server
- Monitor GPU memory with `nvidia-smi`
- The DCAL model is memory-intensive due to attention mechanisms
- Consider using smaller batch sizes if training fails

### **No Internet on Server?** ✅ 
- All dependencies are standard PyTorch/Python packages
- Pre-trained models will be downloaded on first use
- MLFlow runs locally (no external services)
- No wandb or cloud services

### **Memory Considerations**
- **Laptop**: Uses ~500MB-1GB RAM for testing
- **Server**: Uses ~8-10GB GPU memory for full training
- **RTX 2080Ti**: Limited to 11GB - be conservative

### **Common Issues & Solutions**

**"CUDA error" on laptop**: 
- ✅ **Expected behavior** - use `test_laptop.py` instead
- The code is designed to work on both CPU and GPU

**"Killed" process on GPU server**:
- ❌ **GPU memory exhaustion** - RTX 2080Ti out of VRAM
- Use `test_gpu_safe.py` first to check compatibility
- Consider smaller model or batch sizes

**Import errors**:
- Make sure you're in the project root directory
- Check that all files were copied to the server

**Configuration errors**:
- Use `get_auto_config()` for automatic hardware detection
- Specific configs are available for laptop/server if needed

---

## 📁 File Overview

### **Test Scripts**
- `test_laptop.py` - Fast CPU-only testing for development
- `test_gpu_safe.py` - Conservative GPU testing for RTX 2080Ti
- `test_model.py` - Comprehensive testing (adapts to hardware)

### **Key Configuration Files**
- `configs/twin_verification_config.py` - Main configuration
- Hardware auto-detection built-in

### **Model Files**
- `src/models/dcal_verification_model.py` - Main model
- `src/modules/` - Core DCAL components
- All use automatic CPU/GPU detection

---

## 🎯 Recommended Workflow

```bash
# 1. On Laptop (Development)
git clone <your-repo>
cd twin_dcal
python test_laptop.py          # Should pass in ~10 seconds

# 2. Make changes, test iteratively
# ... edit code ...
python test_laptop.py          # Quick validation

# 3. Copy to Server (when ready)
scp -r twin_dcal server:/path/to/
ssh server
cd /path/to/twin_dcal

# 4. On Server (Production) - IMPORTANT ORDER
python test_gpu_safe.py        # Test GPU compatibility FIRST
python test_model.py           # Full testing (if GPU-safe passed)
python train.py                # Start full training
```

---

## ✅ Success Indicators

### **Laptop Testing Success**
```
🎉 ALL LAPTOP TESTS PASSED!
⏱️  Test completed in 8.2 seconds
✅ Your code is ready for the GPU server!
```

### **GPU-Safe Testing Success**
```
🎉 GPU-SAFE TEST PASSED!
⏱️  Test completed in 15.3 seconds
✅ Basic GPU functionality verified!
```

### **Server Testing Success**
```
🖥️  Detected GPU environment: 2 GPU(s)
   GPU 0: NVIDIA GeForce RTX 2080 Ti
   GPU 1: NVIDIA GeForce RTX 2080 Ti
🎉 ALL TESTS PASSED!
   Production environment ready for training.
```

---

## 🆘 Troubleshooting

**Q: Test fails on laptop with CUDA errors**
A: Use `python test_laptop.py` instead of `test_model.py`

**Q: Process gets "killed" on GPU server**
A: RTX 2080Ti ran out of memory. Use `python test_gpu_safe.py` first

**Q: Model too slow on laptop**
A: This is expected - laptop testing is just for validation, not performance

**Q: Server doesn't detect GPUs**
A: Check CUDA installation and driver compatibility

**Q: "GPU memory exhausted" errors**
A: RTX 2080Ti has only 11GB. Use smaller batch sizes or model dimensions

**Q: Import errors after copying to server**
A: Ensure all files copied correctly and you're in the right directory

---

## 🔬 Advanced: RTX 2080Ti Optimization

If you encounter memory issues even with conservative settings:

1. **Reduce model size**:
   ```python
   config.D_MODEL = 384  # Instead of 768
   config.SA_BLOCKS = 6  # Instead of 12
   config.BATCH_SIZE_PER_GPU = 4  # Instead of 8
   ```

2. **Use gradient checkpointing**:
   ```python
   config.GRADIENT_CHECKPOINTING = True
   ```

3. **Monitor memory during training**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

This workflow ensures you can develop efficiently on your laptop while being confident that your code will work correctly on the GPU server, even with RTX 2080Ti memory constraints! 🚀 