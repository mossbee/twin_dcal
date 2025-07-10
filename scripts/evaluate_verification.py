#!/usr/bin/env python3
"""
Evaluation Script for Twin Face Verification using DCAL

This script handles:
1. Loading trained models and test data
2. Computing comprehensive evaluation metrics
3. Generating visualizations and analysis
4. Threshold optimization on validation data
5. Comparison with baseline methods

Usage:
    # Basic evaluation
    python evaluate_verification.py --model checkpoints/best_model.pth --test_data data/test_pairs.json

    # With threshold optimization
    python evaluate_verification.py --model best_model.pth --optimize_threshold --val_data data/val_pairs.json

    # Generate visualizations
    python evaluate_verification.py --model best_model.pth --visualize --output_dir results/
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import torch
from tqdm import tqdm

from utils.twin_inference import TwinInferenceEngine, evaluate_on_test_set
from utils.verification_metrics import VerificationMetrics, evaluate_model_predictions
from utils.twin_visualization import (
    AttentionVisualizer, 
    visualize_verification_results,
    save_verification_examples
)
from configs.twin_verification_config import TwinVerificationConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate DCAL Twin Face Verification Model')
    
    # Model and data paths
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
        '--test_data',
        type=str,
        default='data/test_pairs.json',
        help='Path to test data pairs'
    )
    
    parser.add_argument(
        '--val_data',
        type=str,
        default='data/val_pairs.json',
        help='Path to validation data pairs (for threshold optimization)'
    )
    
    parser.add_argument(
        '--twin_pairs',
        type=str,
        default='data/twin_pairs_infor.json',
        help='Path to twin pairs information'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Verification threshold (if not optimizing)'
    )
    
    parser.add_argument(
        '--optimize_threshold',
        action='store_true',
        help='Optimize threshold on validation data'
    )
    
    parser.add_argument(
        '--threshold_metric',
        type=str,
        default='f1',
        choices=['f1', 'accuracy', 'eer'],
        help='Metric to optimize threshold for'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--save_examples',
        action='store_true',
        help='Save example predictions'
    )
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=50,
        help='Number of examples to save per category'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run evaluation on'
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze_attention',
        action='store_true',
        help='Analyze attention maps'
    )
    
    parser.add_argument(
        '--compare_features',
        action='store_true',
        help='Compare different feature types (SA vs GLCA)'
    )
    
    parser.add_argument(
        '--benchmark_speed',
        action='store_true',
        help='Benchmark inference speed'
    )
    
    return parser.parse_args()


def load_test_pairs(test_data_path: str) -> List[Tuple[str, str, int]]:
    """
    Load test pairs from file
    
    Args:
        test_data_path: Path to test data file
        
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    if test_data_path.endswith('.json'):
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Format: [{"img1": "path1", "img2": "path2", "label": 0/1}, ...]
            pairs = [(item['img1'], item['img2'], item['label']) for item in data]
        elif isinstance(data, dict):
            # Format: {"pairs": [[path1, path2, label], ...]}
            pairs = data.get('pairs', [])
        else:
            raise ValueError(f"Unsupported JSON format in {test_data_path}")
    
    elif test_data_path.endswith('.txt'):
        pairs = []
        with open(test_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img1, img2, label = parts[0], parts[1], int(parts[2])
                    pairs.append((img1, img2, label))
    
    else:
        raise ValueError(f"Unsupported file format: {test_data_path}")
    
    return pairs


def evaluate_model(engine: TwinInferenceEngine,
                  test_pairs: List[Tuple[str, str, int]],
                  twin_pairs_info: List[List[str]],
                  batch_size: int = 16) -> Dict[str, Any]:
    """
    Evaluate model on test data
    
    Args:
        engine: Inference engine
        test_pairs: Test pairs
        twin_pairs_info: Twin pairs information
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation results
    """
    print(f"Evaluating on {len(test_pairs)} test pairs...")
    
    # Extract image pairs and labels
    image_pairs = [(pair[0], pair[1]) for pair in test_pairs]
    labels = np.array([pair[2] for pair in test_pairs])
    paths = [(pair[0], pair[1]) for pair in test_pairs]
    
    # Get predictions
    start_time = time.time()
    results = engine.batch_verify(image_pairs, batch_size, show_progress=True)
    inference_time = time.time() - start_time
    
    predictions = np.array([r['verification_score'] for r in results])
    
    # Compute metrics
    metrics_calculator = VerificationMetrics(twin_pairs_info)
    metrics = metrics_calculator.compute_all_metrics(predictions, labels, paths)
    
    # Add timing information
    metrics['inference_time_total'] = inference_time
    metrics['inference_time_per_pair'] = inference_time / len(test_pairs)
    metrics['throughput_pairs_per_second'] = len(test_pairs) / inference_time
    
    # Create model outputs dictionary for other functions
    model_outputs = {
        'predictions': predictions,
        'labels': labels,
        'paths': paths
    }
    
    return {
        'metrics': metrics,
        'model_outputs': model_outputs
    }


def optimize_threshold(engine: TwinInferenceEngine,
                      val_pairs: List[Tuple[str, str, int]],
                      metric: str = 'f1') -> float:
    """
    Optimize verification threshold on validation data
    
    Args:
        engine: Inference engine
        val_pairs: Validation pairs
        metric: Metric to optimize
        
    Returns:
        Optimal threshold
    """
    print(f"Optimizing threshold on {len(val_pairs)} validation pairs...")
    
    optimal_threshold = engine.optimize_threshold(val_pairs, metric)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold


def analyze_attention_maps(engine: TwinInferenceEngine,
                          test_pairs: List[Tuple[str, str, int]],
                          output_dir: Path,
                          num_samples: int = 20):
    """
    Analyze attention maps for sample pairs
    
    Args:
        engine: Inference engine
        test_pairs: Test pairs
        output_dir: Output directory
        num_samples: Number of samples to analyze
    """
    print(f"Analyzing attention maps for {num_samples} samples...")
    
    attention_dir = output_dir / 'attention_analysis'
    attention_dir.mkdir(exist_ok=True)
    
    # Sample pairs from different categories
    positive_pairs = [pair for pair in test_pairs if pair[2] == 1]
    negative_pairs = [pair for pair in test_pairs if pair[2] == 0]
    
    samples_per_category = num_samples // 2
    sampled_pairs = (
        positive_pairs[:samples_per_category] + 
        negative_pairs[:samples_per_category]
    )
    
    visualizer = AttentionVisualizer(save_dir=str(attention_dir))
    
    for i, (img1_path, img2_path, label) in enumerate(tqdm(sampled_pairs, desc="Processing attention")):
        try:
            # Get prediction with attention
            result = engine.verify_pair(
                img1_path, img2_path, 
                return_attention=True
            )
            
            if 'attention_maps' in result:
                attention_maps1 = {}
                attention_maps2 = {}
                
                for key, value in result['attention_maps'].items():
                    if key.endswith('1'):
                        attention_maps1[key[:-1]] = value
                    elif key.endswith('2'):
                        attention_maps2[key[:-1]] = value
                
                # Create comparison visualization
                save_path = attention_dir / f'attention_comparison_{i:03d}_gt_{label}_pred_{result["verification_score"]:.3f}.png'
                
                fig = visualizer.visualize_verification_comparison(
                    img1_path, img2_path,
                    result['verification_score'], label,
                    attention_maps1, attention_maps2,
                    str(save_path)
                )
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
    
    print(f"Attention analysis saved to {attention_dir}")


def compare_feature_types(engine: TwinInferenceEngine,
                         test_pairs: List[Tuple[str, str, int]],
                         output_dir: Path,
                         num_samples: int = 1000):
    """
    Compare different feature types (SA vs GLCA vs Combined)
    
    Args:
        engine: Inference engine
        test_pairs: Test pairs
        output_dir: Output directory
        num_samples: Number of samples to analyze
    """
    print(f"Comparing feature types on {num_samples} samples...")
    
    # Sample test pairs
    sampled_pairs = test_pairs[:num_samples]
    labels = np.array([pair[2] for pair in sampled_pairs])
    
    feature_comparison = {}
    
    for feature_type in ['sa', 'glca', 'combined']:
        print(f"Extracting {feature_type} features...")
        
        # Extract features
        images = []
        for img1_path, img2_path, _ in sampled_pairs:
            images.extend([img1_path, img2_path])
        
        features = engine.extract_features(images, feature_type=feature_type)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(0, len(features), 2):
            feat1 = features[i]
            feat2 = features[i + 1]
            
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_acc = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            binary_pred = (similarities >= threshold).astype(int)
            acc = accuracy_score(labels, binary_pred)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        roc_auc = roc_auc_score(labels, similarities)
        
        feature_comparison[feature_type] = {
            'roc_auc': roc_auc,
            'best_accuracy': best_acc,
            'best_threshold': best_threshold,
            'similarities': similarities
        }
        
        print(f"  {feature_type}: ROC AUC={roc_auc:.4f}, Best Acc={best_acc:.4f}")
    
    # Save comparison results
    comparison_file = output_dir / 'feature_comparison.json'
    with open(comparison_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        comparison_data = {}
        for feature_type, data in feature_comparison.items():
            comparison_data[feature_type] = {
                'roc_auc': float(data['roc_auc']),
                'best_accuracy': float(data['best_accuracy']),
                'best_threshold': float(data['best_threshold'])
            }
        json.dump(comparison_data, f, indent=2)
    
    print(f"Feature comparison saved to {comparison_file}")
    return feature_comparison


def benchmark_inference_speed(engine: TwinInferenceEngine,
                             output_dir: Path):
    """
    Benchmark inference speed
    
    Args:
        engine: Inference engine
        output_dir: Output directory
    """
    print("Benchmarking inference speed...")
    
    from utils.twin_inference import benchmark_inference_speed
    
    results = benchmark_inference_speed(engine.model_path)
    
    # Save results
    benchmark_file = output_dir / 'speed_benchmark.json'
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Speed benchmark results:")
    for batch_size, avg_time in results.items():
        throughput = batch_size / avg_time
        print(f"  Batch size {batch_size}: {avg_time:.4f}s ({throughput:.2f} pairs/s)")
    
    print(f"Benchmark results saved to {benchmark_file}")


def save_evaluation_report(metrics: Dict[str, float],
                          optimal_threshold: float,
                          output_dir: Path):
    """
    Save comprehensive evaluation report
    
    Args:
        metrics: Evaluation metrics
        optimal_threshold: Optimal threshold
        output_dir: Output directory
    """
    report = {
        'evaluation_summary': {
            'optimal_threshold': optimal_threshold,
            'verification_accuracy': metrics.get('verification_accuracy', 0.0),
            'roc_auc': metrics.get('roc_auc', 0.0),
            'eer': metrics.get('eer', 0.0),
            'pr_auc': metrics.get('pr_auc', 0.0),
            'tar_at_far_0.01': metrics.get('tar_at_far_0.01', 0.0),
            'tar_at_far_0.001': metrics.get('tar_at_far_0.001', 0.0)
        },
        'detailed_metrics': metrics,
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'threshold_optimization': {
            'optimized': optimal_threshold != 0.5,
            'value': optimal_threshold
        }
    }
    
    # Add twin-specific metrics if available
    twin_metrics = {key: value for key, value in metrics.items() if 'twin' in key}
    if twin_metrics:
        report['twin_analysis'] = twin_metrics
    
    # Save report
    report_file = output_dir / 'evaluation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary text file
    summary_file = output_dir / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("DCAL Twin Face Verification - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Evaluation Date: {report['evaluation_date']}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
        
        f.write("Key Metrics:\n")
        f.write(f"  Verification Accuracy: {metrics.get('verification_accuracy', 0.0):.4f}\n")
        f.write(f"  ROC AUC: {metrics.get('roc_auc', 0.0):.4f}\n")
        f.write(f"  Equal Error Rate: {metrics.get('eer', 0.0):.4f}\n")
        f.write(f"  PR AUC: {metrics.get('pr_auc', 0.0):.4f}\n")
        f.write(f"  TAR@FAR=0.01: {metrics.get('tar_at_far_0.01', 0.0):.4f}\n")
        f.write(f"  TAR@FAR=0.001: {metrics.get('tar_at_far_0.001', 0.0):.4f}\n\n")
        
        if twin_metrics:
            f.write("Twin Analysis:\n")
            for key, value in twin_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nTiming:\n")
        f.write(f"  Total inference time: {metrics.get('inference_time_total', 0.0):.2f}s\n")
        f.write(f"  Time per pair: {metrics.get('inference_time_per_pair', 0.0):.4f}s\n")
        f.write(f"  Throughput: {metrics.get('throughput_pairs_per_second', 0.0):.2f} pairs/s\n")
    
    print(f"Evaluation report saved to {report_file}")
    print(f"Summary saved to {summary_file}")


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"DCAL Twin Face Verification - Evaluation")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load inference engine
    print("Loading model...")
    engine = TwinInferenceEngine(
        args.model, 
        args.config, 
        args.device, 
        args.threshold
    )
    
    # Load test data
    print("Loading test data...")
    test_pairs = load_test_pairs(args.test_data)
    print(f"Loaded {len(test_pairs)} test pairs")
    
    # Load twin pairs info
    twin_pairs_info = []
    if os.path.exists(args.twin_pairs):
        with open(args.twin_pairs, 'r') as f:
            twin_pairs_info = json.load(f)
        print(f"Loaded {len(twin_pairs_info)} twin pairs")
    
    # Optimize threshold if requested
    optimal_threshold = args.threshold
    if args.optimize_threshold:
        print("Loading validation data for threshold optimization...")
        val_pairs = load_test_pairs(args.val_data)
        optimal_threshold = optimize_threshold(engine, val_pairs, args.threshold_metric)
        engine.set_threshold(optimal_threshold)
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(engine, test_pairs, twin_pairs_info, args.batch_size)
    metrics = eval_results['metrics']
    model_outputs = eval_results['model_outputs']
    
    # Print key results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Verification Accuracy: {metrics['verification_accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Equal Error Rate: {metrics['eer']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"TAR@FAR=0.01: {metrics.get('tar_at_far_0.01', 0.0):.4f}")
    print(f"Throughput: {metrics['throughput_pairs_per_second']:.2f} pairs/s")
    
    # Twin-specific results
    if 'twin_accuracy' in metrics:
        print(f"\nTwin Analysis:")
        print(f"Twin Accuracy: {metrics['twin_accuracy']:.4f}")
        print(f"Regular Accuracy: {metrics.get('regular_accuracy', 0.0):.4f}")
        print(f"Twin Samples: {metrics.get('twin_samples', 0)}")
    
    # Save evaluation report
    save_evaluation_report(metrics, optimal_threshold, output_dir)
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        figures = visualize_verification_results(
            model_outputs['predictions'],
            model_outputs['labels'],
            str(output_dir / 'visualizations')
        )
        print(f"Visualizations saved to {output_dir / 'visualizations'}")
    
    # Save example predictions
    if args.save_examples:
        print("\nSaving example predictions...")
        save_verification_examples(
            model_outputs,
            None,  # We don't have the dataset object
            str(output_dir / 'examples'),
            args.num_examples
        )
        print(f"Examples saved to {output_dir / 'examples'}")
    
    # Additional analyses
    if args.analyze_attention:
        analyze_attention_maps(engine, test_pairs, output_dir)
    
    if args.compare_features:
        compare_feature_types(engine, test_pairs, output_dir)
    
    if args.benchmark_speed:
        benchmark_inference_speed(engine, output_dir)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main() 