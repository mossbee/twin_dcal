#!/usr/bin/env python3
"""
Demo Script for Twin Face Verification using DCAL

This script provides an interactive demo for testing the DCAL model on face verification tasks.
Supports both single pair verification and batch processing.

Usage:
    # Interactive demo
    python demo.py --model checkpoints/best_model.pth

    # Single pair verification
    python demo.py --model best_model.pth --img1 face1.jpg --img2 face2.jpg

    # Batch verification from folder
    python demo.py --model best_model.pth --demo_folder demo_images/
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils.twin_inference import TwinInferenceEngine
from utils.twin_visualization import AttentionVisualizer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DCAL Twin Face Verification Demo')
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda/cpu)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Verification threshold'
    )
    
    # Demo modes
    parser.add_argument(
        '--img1',
        type=str,
        default=None,
        help='Path to first image for single pair verification'
    )
    
    parser.add_argument(
        '--img2',
        type=str,
        default=None,
        help='Path to second image for single pair verification'
    )
    
    parser.add_argument(
        '--demo_folder',
        type=str,
        default=None,
        help='Folder containing demo image pairs'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive demo mode'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='demo_results',
        help='Directory to save demo results'
    )
    
    parser.add_argument(
        '--show_attention',
        action='store_true',
        help='Show attention maps in results'
    )
    
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save demo results to files'
    )
    
    return parser.parse_args()


def load_and_display_image(image_path: str, title: str = ""):
    """
    Load and display an image
    
    Args:
        image_path: Path to image file
        title: Title for the image
        
    Returns:
        PIL Image object
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def verify_single_pair(engine: TwinInferenceEngine,
                      img1_path: str,
                      img2_path: str,
                      show_attention: bool = False,
                      save_dir: str = None) -> dict:
    """
    Verify a single pair of images
    
    Args:
        engine: Inference engine
        img1_path: Path to first image
        img2_path: Path to second image
        show_attention: Whether to return attention maps
        save_dir: Directory to save results
        
    Returns:
        Verification results
    """
    print(f"\nVerifying pair:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    
    # Check if images exist
    if not os.path.exists(img1_path):
        print(f"Error: Image 1 not found: {img1_path}")
        return None
    
    if not os.path.exists(img2_path):
        print(f"Error: Image 2 not found: {img2_path}")
        return None
    
    # Perform verification
    start_time = time.time()
    result = engine.verify_pair(
        img1_path, img2_path,
        return_attention=show_attention,
        return_features=True
    )
    inference_time = time.time() - start_time
    
    # Display results
    score = result['verification_score']
    decision = result['verification_decision']
    confidence = result.get('confidence', 0.0)
    
    print(f"\n" + "="*50)
    print(f"VERIFICATION RESULTS")
    print(f"="*50)
    print(f"Similarity Score: {score:.4f}")
    print(f"Decision: {'MATCH' if decision else 'NO MATCH'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference Time: {inference_time:.4f}s")
    
    # Create visualization
    if show_attention and 'attention_maps' in result:
        print(f"\nGenerating attention visualization...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            visualizer = AttentionVisualizer(save_dir=save_dir)
            
            # Extract attention maps for each image
            attention_maps1 = {}
            attention_maps2 = {}
            
            for key, value in result['attention_maps'].items():
                if key.endswith('_1'):
                    attention_maps1[key[:-2]] = value
                elif key.endswith('_2'):
                    attention_maps2[key[:-2]] = value
            
            # Create visualization
            save_path = os.path.join(save_dir, f'verification_result_score_{score:.3f}.png')
            visualizer.visualize_verification_comparison(
                img1_path, img2_path,
                score, decision,
                attention_maps1, attention_maps2,
                save_path
            )
            
            print(f"Attention visualization saved to: {save_path}")
    
    # Add timing info to result
    result['inference_time'] = inference_time
    
    return result


def demo_folder_pairs(engine: TwinInferenceEngine,
                     demo_folder: str,
                     output_dir: str = None,
                     show_attention: bool = False):
    """
    Demo verification on pairs of images in a folder
    
    Args:
        engine: Inference engine
        demo_folder: Folder containing demo images
        output_dir: Directory to save results
        show_attention: Whether to show attention maps
    """
    demo_folder = Path(demo_folder)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(demo_folder.glob(f'*{ext}')))
        image_files.extend(list(demo_folder.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {demo_folder}")
        return
    
    print(f"Found {len(image_files)} images in {demo_folder}")
    
    # Create all possible pairs
    results = []
    total_pairs = len(image_files) * (len(image_files) - 1) // 2
    
    print(f"Testing {total_pairs} image pairs...")
    
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            img1_path = str(image_files[i])
            img2_path = str(image_files[j])
            
            # Create output directory for this pair
            pair_output_dir = None
            if output_dir:
                pair_name = f"pair_{i:02d}_{j:02d}_{image_files[i].stem}_{image_files[j].stem}"
                pair_output_dir = os.path.join(output_dir, pair_name)
            
            # Verify pair
            result = verify_single_pair(
                engine, img1_path, img2_path,
                show_attention, pair_output_dir
            )
            
            if result:
                result['img1_path'] = img1_path
                result['img2_path'] = img2_path
                result['img1_name'] = image_files[i].name
                result['img2_name'] = image_files[j].name
                results.append(result)
    
    # Summary statistics
    if results:
        scores = [r['verification_score'] for r in results]
        matches = [r['verification_decision'] for r in results]
        
        print(f"\n" + "="*60)
        print(f"DEMO SUMMARY")
        print(f"="*60)
        print(f"Total pairs tested: {len(results)}")
        print(f"Matches found: {sum(matches)}")
        print(f"Average similarity: {np.mean(scores):.4f}")
        print(f"Max similarity: {max(scores):.4f}")
        print(f"Min similarity: {min(scores):.4f}")
        
        # Show best matches
        sorted_results = sorted(results, key=lambda x: x['verification_score'], reverse=True)
        
        print(f"\nTop 5 most similar pairs:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['img1_name']} vs {result['img2_name']}: {result['verification_score']:.4f}")


def interactive_demo(engine: TwinInferenceEngine,
                    output_dir: str = None,
                    show_attention: bool = False):
    """
    Run interactive demo mode
    
    Args:
        engine: Inference engine
        output_dir: Directory to save results
        show_attention: Whether to show attention maps
    """
    print("\n" + "="*60)
    print("DCAL TWIN FACE VERIFICATION - INTERACTIVE DEMO")
    print("="*60)
    print("Commands:")
    print("  verify <img1> <img2>  - Verify two images")
    print("  folder <path>         - Demo on folder of images")
    print("  threshold <value>     - Set verification threshold")
    print("  help                  - Show this help")
    print("  quit                  - Exit demo")
    print("="*60)
    
    while True:
        try:
            # Get user input
            command = input("\nDemo> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("Goodbye!")
                break
            
            elif cmd == 'help':
                print("\nCommands:")
                print("  verify <img1> <img2>  - Verify two images")
                print("  folder <path>         - Demo on folder of images")
                print("  threshold <value>     - Set verification threshold")
                print("  help                  - Show this help")
                print("  quit                  - Exit demo")
            
            elif cmd == 'verify':
                if len(command) != 3:
                    print("Usage: verify <img1> <img2>")
                    continue
                
                img1_path, img2_path = command[1], command[2]
                
                result = verify_single_pair(
                    engine, img1_path, img2_path,
                    show_attention, output_dir
                )
            
            elif cmd == 'folder':
                if len(command) != 2:
                    print("Usage: folder <path>")
                    continue
                
                folder_path = command[1]
                
                if not os.path.exists(folder_path):
                    print(f"Error: Folder not found: {folder_path}")
                    continue
                
                demo_folder_pairs(engine, folder_path, output_dir, show_attention)
            
            elif cmd == 'threshold':
                if len(command) != 2:
                    print("Usage: threshold <value>")
                    continue
                
                try:
                    new_threshold = float(command[1])
                    engine.set_threshold(new_threshold)
                    print(f"Verification threshold set to: {new_threshold}")
                except ValueError:
                    print("Error: Invalid threshold value")
            
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demo function"""
    args = parse_arguments()
    
    print("DCAL Twin Face Verification Demo")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    
    # Create output directory
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    else:
        output_dir = None
    
    # Load model
    print("\nLoading model...")
    try:
        engine = TwinInferenceEngine(
            args.model,
            args.config,
            args.device,
            args.threshold
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Determine demo mode
    if args.img1 and args.img2:
        # Single pair verification
        result = verify_single_pair(
            engine, args.img1, args.img2,
            args.show_attention,
            str(output_dir) if output_dir else None
        )
    
    elif args.demo_folder:
        # Batch demo on folder
        demo_folder_pairs(
            engine, args.demo_folder,
            str(output_dir) if output_dir else None,
            args.show_attention
        )
    
    elif args.interactive:
        # Interactive mode
        interactive_demo(
            engine,
            str(output_dir) if output_dir else None,
            args.show_attention
        )
    
    else:
        # Default to interactive mode
        print("\nNo specific demo mode specified. Starting interactive demo...")
        interactive_demo(
            engine,
            str(output_dir) if output_dir else None,
            args.show_attention
        )


if __name__ == '__main__':
    main() 