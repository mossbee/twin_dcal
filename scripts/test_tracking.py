#!/usr/bin/env python3
"""
Test Tracking Configuration Script

This script helps you verify that your tracking setup (MLFlow, WandB, or none) 
is working correctly before starting full training.

Usage:
    python scripts/test_tracking.py --tracking mlflow --mlflow_uri "http://your-server:5000"
    python scripts/test_tracking.py --tracking wandb --wandb_entity "your-username"
    python scripts/test_tracking.py --tracking none
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from configs.twin_verification_config import TwinVerificationConfig


def test_mlflow(config):
    """Test MLFlow connection"""
    print("🔍 Testing MLFlow connection...")
    
    try:
        import mlflow
        print("✅ MLFlow package imported successfully")
        
        # Set tracking URI
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        print(f"✅ MLFlow tracking URI set: {config.MLFLOW_TRACKING_URI}")
        
        # Try to create/get experiment
        experiment_name = config.MLFLOW_EXPERIMENT_NAME
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                print(f"✅ Found existing experiment: {experiment_name}")
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"⚠️  Could not access experiment, but will be created during training: {e}")
        
        # Test run creation
        with mlflow.start_run(run_name="test_run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.5)
            print(f"✅ Successfully created test run: {run.info.run_id}")
        
        print("🎉 MLFlow configuration is working correctly!")
        return True
        
    except ImportError:
        print("❌ MLFlow not installed. Install with: pip install mlflow")
        return False
    except Exception as e:
        print(f"❌ MLFlow connection failed: {e}")
        print("   Check your MLFLOW_TRACKING_URI and ensure the server is running")
        return False


def test_wandb(config):
    """Test WandB connection"""
    print("🔍 Testing WandB connection...")
    
    try:
        import wandb
        print("✅ WandB package imported successfully")
        
        # Check API key
        api_key = os.environ.get('WANDB_API_KEY')
        if not api_key:
            print("⚠️  WANDB_API_KEY not set. Run 'wandb login' or set the environment variable")
        else:
            print("✅ WANDB_API_KEY found in environment")
        
        # Test initialization
        wandb_config = {
            "test_param": "test_value",
            "tracking_test": True
        }
        
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name="test_run",
            tags=["test"] + config.WANDB_TAGS,
            config=wandb_config,
            mode="online"  # Force online mode to test connection
        )
        
        # Log test metrics
        wandb.log({"test_metric": 0.5, "step": 1})
        print(f"✅ Successfully created WandB run: {run.name}")
        
        # Finish run
        wandb.finish()
        
        print("🎉 WandB configuration is working correctly!")
        return True
        
    except ImportError:
        print("❌ WandB not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"❌ WandB connection failed: {e}")
        print("   Check your WANDB_API_KEY and WANDB_ENTITY settings")
        return False


def test_no_tracking():
    """Test no tracking mode"""
    print("🔍 Testing no tracking mode...")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard available for local logging")
        
        # Test TensorBoard writer
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SummaryWriter(log_dir=temp_dir)
            writer.add_scalar('test_metric', 0.5, 1)
            writer.close()
            print("✅ TensorBoard writer test successful")
        
        print("🎉 No tracking mode is working correctly!")
        print("   Only TensorBoard will be used for local logging")
        return True
        
    except ImportError:
        print("❌ TensorBoard not available. Install with: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ TensorBoard test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test tracking configuration')
    
    parser.add_argument(
        '--tracking',
        type=str,
        choices=['mlflow', 'wandb', 'none'],
        required=True,
        help='Tracking method to test'
    )
    
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default=None,
        help='MLFlow server URI override'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='WandB project name override'
    )
    
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='WandB entity override'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRACKING CONFIGURATION TEST")
    print("=" * 60)
    
    # Create config
    config = TwinVerificationConfig()
    
    # Apply overrides
    if args.mlflow_uri:
        config.MLFLOW_TRACKING_URI = args.mlflow_uri
    if args.wandb_project:
        config.WANDB_PROJECT = args.wandb_project
    if args.wandb_entity:
        config.WANDB_ENTITY = args.wandb_entity
    
    # Test based on tracking method
    success = False
    
    if args.tracking == 'mlflow':
        print(f"Testing MLFlow with URI: {config.MLFLOW_TRACKING_URI}")
        print(f"Experiment name: {config.MLFLOW_EXPERIMENT_NAME}")
        success = test_mlflow(config)
        
    elif args.tracking == 'wandb':
        print(f"Testing WandB with project: {config.WANDB_PROJECT}")
        print(f"Entity: {config.WANDB_ENTITY}")
        success = test_wandb(config)
        
    elif args.tracking == 'none':
        print("Testing no tracking mode (TensorBoard only)")
        success = test_no_tracking()
    
    print("=" * 60)
    
    if success:
        print("✅ CONFIGURATION TEST PASSED")
        print(f"   You can now run training with --tracking {args.tracking}")
        
        # Show example command
        print("\nExample training command:")
        if args.tracking == 'mlflow':
            cmd = f"python scripts/train_twin_verification.py --config debug --tracking mlflow"
            if args.mlflow_uri:
                cmd += f" --mlflow_uri \"{args.mlflow_uri}\""
        elif args.tracking == 'wandb':
            cmd = f"python scripts/train_twin_verification.py --config debug --tracking wandb"
            if args.wandb_entity:
                cmd += f" --wandb_entity \"{args.wandb_entity}\""
            if args.wandb_project:
                cmd += f" --wandb_project \"{args.wandb_project}\""
        else:
            cmd = "python scripts/train_twin_verification.py --config debug --tracking none"
        
        print(f"   {cmd}")
        
    else:
        print("❌ CONFIGURATION TEST FAILED")
        print("   Please fix the issues above before running training")
        sys.exit(1)


if __name__ == '__main__':
    main() 