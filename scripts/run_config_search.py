#!/usr/bin/env python3
"""
Simple script to run automated configuration search in Kaggle notebooks

Usage:
    # In Kaggle notebook cell:
    !python scripts/run_config_search.py
    
    # Or with specific parameters:
    !python scripts/run_config_search.py --test_time 180 --gpu_type p100
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Change to project directory
    os.chdir('/kaggle/working/twin_dcal')
    
    # Set up WandB API key from Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        print("✅ WandB API key configured")
    except Exception as e:
        print(f"⚠️  Warning: Could not set WandB API key: {e}")
        print("   Make sure WANDB_API_KEY is in your Kaggle secrets")
    
    # Detect GPU type
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_count = torch.cuda.device_count()
            
            if "P100" in gpu_name:
                gpu_type = "p100"
                print(f"🔥 Detected Kaggle P100 GPU")
            elif "T4" in gpu_name and gpu_count >= 2:
                gpu_type = "t4_distributed"
                print(f"🔥 Detected Kaggle T4 x{gpu_count} GPUs")
            else:
                gpu_type = "p100"  # Default fallback
                print(f"🔥 Detected {gpu_name} (using P100 config as fallback)")
        else:
            print("❌ No GPU detected")
            return
    except ImportError:
        gpu_type = "p100"
        print("⚠️  Could not detect GPU type, using P100 config")
    
    # Run automated search
    cmd = [
        sys.executable, "scripts/automated_config_search.py",
        "--gpu_type", gpu_type,
        "--test_duration", "180",  # 3 minutes per config
        "--wandb_entity", "hunchoquavodb-hanoi-university-of-science-and-technology"
    ]
    
    print(f"🚀 Running automated configuration search...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   GPU type: {gpu_type}")
    print(f"   Test duration: 3 minutes per config")
    print(f"   This may take 15-30 minutes total")
    print()
    
    # Run the search
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # Check if results were generated
        if os.path.exists("configs/search_results/best_config.json"):
            print("\n🎉 Configuration search completed successfully!")
            print("📁 Results saved to: configs/search_results/")
            print("🚀 You can now run training with the best config:")
            print("   !python scripts/train_twin_verification.py --config search_best --wandb_entity 'hunchoquavodb-hanoi-university-of-science-and-technology' --wandb_project 'dcal-twin-verification'")
        else:
            print("\n⚠️  Search completed but no working config was found")
            print("   Check the output above for errors")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Configuration search failed with exit code {e.returncode}")
        print("   Check the output above for errors")
    except KeyboardInterrupt:
        print("\n⏸️  Search interrupted by user")

if __name__ == "__main__":
    main() 