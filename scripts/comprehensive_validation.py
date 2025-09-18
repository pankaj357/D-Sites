#!/usr/bin/env python3
"""
comprehensive_validation.py - Publication-ready validation for D-Sites TFBS predictor

Performs:
1. K-fold cross-validation for robust performance estimates
2. Multiple TF evaluation to show generalizability
3. Feature ablation study to quantify component contributions
4. Statistical power analysis with effect sizes
5. Comparison with additional tools beyond FIMO

Usage:
python comprehensive_validation.py --known collectf_export.tsv --fasta genome.fasta 
                                  --gff annotation.gff --motif_dir motifs/ 
                                  --tfs AraC,LexA,Crp,Fur --out_dir validation_results/
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from tqdm import tqdm
import sys
import subprocess
from pathlib import Path

# Add your tool's directory to path
sys.path.append('/path/to/your/dsites/tool')

# Import your tool's functions (adjust based on your actual implementation)
from your_tool_module import run_dsites, train_model, predict_sites

def load_known_sites(known_path, tf_filter=None):
    """Load and filter known binding sites"""
    known_df = pd.read_csv(known_path, sep='\t')
    if tf_filter:
        known_df = known_df[known_df['TF'] == tf_filter]
    return known_df

def run_kfold_cross_validation(known_df, genome_fasta, motif_path, gff_path, n_folds=5):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(known_df)):
        print(f"Running fold {fold + 1}/{n_folds}...")
        
        train_sites = known_df.iloc[train_idx]
        test_sites = known_df.iloc[test_idx]
        
        # Train model on training sites
        model = train_model(train_sites, genome_fasta, motif_path)
        
        # Predict on test sites
        predictions = predict_sites(model, genome_fasta, gff_path)
        
        # Evaluate performance
        metrics = evaluate_predictions(predictions, test_sites)
        metrics['fold'] = fold + 1
        fold_results.append(metrics)
    
    return pd.DataFrame(fold_results)

def evaluate_multiple_tfs(tf_list, known_path, genome_fasta, motif_dir, gff_path):
    """Benchmark across multiple transcription factors"""
    tf_results = {}
    
    for tf in tqdm(tf_list, desc="Evaluating TFs"):
        print(f"\nEvaluating {tf}...")
        
        # Load motif for this TF
        motif_path = os.path.join(motif_dir, f"{tf}.meme")
        if not os.path.exists(motif_path):
            print(f"Motif not found for {tf}, skipping...")
            continue
        
        # Filter known sites for this TF
        known_df = load_known_sites(known_path, tf)
        if len(known_df) < 10:  # Minimum sites for meaningful evaluation
            print(f"Insufficient known sites for {tf}, skipping...")
            continue
        
        # Run 5-fold CV for this TF
        cv_results = run_kfold_cross_validation(known_df, genome_fasta, motif_path, gff_path)
        
        # Store results
        tf_results[tf] = {
            'mean_precision': cv_results['precision'].mean(),
            'mean_recall': cv_results['recall'].mean(),
            'mean_f1': cv_results['f1_score'].mean(),
            'mean_auc_pr': cv_results['auc_pr'].mean(),
            'n_sites': len(known_df),
            'cv_results': cv_results.to_dict()
        }
    
    return tf_results

def feature_ablation_study(known_df, genome_fasta, motif_path, gff_path):
    """Test contribution of each feature component"""
    ablation_results = {}
    
    # Full model (PWM + Shape + RF)
    print("Running full model...")
    full_metrics = run_with_configuration(known_df, genome_fasta, motif_path, gff_path, 
                                        use_pwm=True, use_shape=True, use_rf=True)
    ablation_results['full_model'] = full_metrics
    
    # PWM + Shape only (no RF)
    print("Running PWM + Shape only...")
    pwm_shape_metrics = run_with_configuration(known_df, genome_fasta, motif_path, gff_path,
                                             use_pwm=True, use_shape=True, use_rf=False)
    ablation_results['pwm_shape'] = pwm_shape_metrics
    
    # PWM only
    print("Running PWM only...")
    pwm_only_metrics = run_with_configuration(known_df, genome_fasta, motif_path, gff_path,
                                            use_pwm=True, use_shape=False, use_rf=False)
    ablation_results['pwm_only'] = pwm_only_metrics
    
    # RF only (no sequence features)
    print("Running RF only...")
    rf_only_metrics = run_with_configuration(known_df, genome_fasta, motif_path, gff_path,
                                           use_pwm=False, use_shape=False, use_rf=True)
    ablation_results['rf_only'] = rf_only_metrics
    
    return pd.DataFrame(ablation_results)

def run_with_configuration(known_df, genome_fasta, motif_path, gff_path, 
                         use_pwm=True, use_shape=True, use_rf=True):
    """Run tool with specific feature configuration"""
    # This function should interface with your tool's modified version
    # that can toggle different features on/off
    
    # For now, placeholder implementation
    # You'll need to modify your tool to accept these parameters
    config = {
        'use_pwm': use_pwm,
        'use_shape': use_shape, 
        'use_rf': use_rf
    }
    
    # Run your tool with this configuration
    predictions = run_dsites(genome_fasta, motif_path, gff_path, config=config)
    
    # Evaluate predictions
    return evaluate_predictions(predictions, known_df)

def compare_with_additional_tools(known_df, genome_fasta, motif_path, gff_path):
    """Compare with multiple established tools"""
    tool_results = {}
    
    # Your tool (D-Sites)
    print("Running D-Sites...")
    dsites_pred = run_dsites(genome_fasta, motif_path, gff_path)
    tool_results['D-Sites'] = evaluate_predictions(dsites_pred, known_df)
    
    # FIMO
    print("Running FIMO...")
    fimo_pred = run_fimo(genome_fasta, motif_path)
    tool_results['FIMO'] = evaluate_predictions(fimo_pred, known_df)
    
    # MEME (if available)
    if tool_available('meme'):
        print("Running MEME...")
        meme_pred = run_meme(genome_fasta, motif_path)
        tool_results['MEME'] = evaluate_predictions(meme_pred, known_df)
    
    # HOMER (if available)
    if tool_available('homer'):
        print("Running HOMER...")
        homer_pred = run_homer(genome_fasta, motif_path)
        tool_results['HOMER'] = evaluate_predictions(homer_pred, known_df)
    
    return pd.DataFrame(tool_results)

def run_fimo(genome_fasta, motif_path):
    """Run FIMO and parse results"""
    # Implement FIMO execution and result parsing
    pass

def run_meme(genome_fasta, motif_path):
    """Run MEME and parse results"""
    # Implement MEME execution and result parsing
    pass

def run_homer(genome_fasta, motif_path):
    """Run HOMER and parse results"""
    # Implement HOMER execution and result parsing
    pass

def tool_available(tool_name):
    """Check if a tool is available in PATH"""
    return subprocess.run(['which', tool_name], capture_output=True).returncode == 0

def evaluate_predictions(predictions, known_sites):
    """Comprehensive evaluation metrics"""
    # Convert to interval trees for efficient overlap checking
    pred_tree = IntervalTree()
    known_tree = IntervalTree()
    
    for _, pred in predictions.iterrows():
        pred_tree.addi(pred['start'], pred['end'] + 1, pred['score'])
    
    for _, known in known_sites.iterrows():
        known_tree.addi(known['site_start'], known['site_end'] + 1)
    
    # Calculate metrics
    metrics = {}
    
    # Precision, Recall, F1
    tp = fp = fn = 0
    
    # True positives: predictions overlapping known sites
    for pred_iv in pred_tree:
        if known_tree.overlaps(pred_iv.begin, pred_iv.end):
            tp += 1
        else:
            fp += 1
    
    # False negatives: known sites not hit by any prediction
    for known_iv in known_tree:
        if not pred_tree.overlaps(known_iv.begin, known_iv.end):
            fn += 1
    
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # AUC calculations
    y_true, y_score = [], []
    for pred_iv in pred_tree:
        y_true.append(1 if known_tree.overlaps(pred_iv.begin, pred_iv.end) else 0)
        y_score.append(pred_iv.data)
    
    if len(set(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['auc_pr'] = auc(recall, precision)
        metrics['auc_roc'] = roc_auc_score(y_true, y_score)
    else:
        metrics['auc_pr'] = metrics['auc_roc'] = 0
    
    return metrics

def statistical_power_analysis(known_df, genome_fasta, motif_path, gff_path, n_permutations=1000):
    """Compute statistical power and effect sizes"""
    # Run actual predictions
    actual_pred = run_dsites(genome_fasta, motif_path, gff_path)
    actual_metrics = evaluate_predictions(actual_pred, known_df)
    
    # Permutation tests for null distribution
    null_metrics = []
    for i in tqdm(range(n_permutations), desc="Permutation tests"):
        # Shuffle known sites to create null distribution
        shuffled_known = known_df.copy()
        shuffled_known['site_start'] = np.random.permutation(shuffled_known['site_start'])
        shuffled_known['site_end'] = shuffled_known['site_start'] + \
                                   (shuffled_known['site_end'] - shuffled_known['site_start'])
        
        null_pred = run_dsites(genome_fasta, motif_path, gff_path)
        null_metrics.append(evaluate_predictions(null_pred, shuffled_known))
    
    null_df = pd.DataFrame(null_metrics)
    
    # Calculate effect sizes and power
    power_analysis = {}
    for metric in ['precision', 'recall', 'f1_score', 'auc_pr']:
        actual_value = actual_metrics[metric]
        null_values = null_df[metric]
        
        # Effect size (Cohen's d)
        effect_size = (actual_value - null_values.mean()) / null_values.std()
        
        # Statistical power
        alpha = 0.05
        critical_value = null_values.quantile(1 - alpha)
        power = (null_values >= critical_value).mean()
        
        power_analysis[metric] = {
            'actual': actual_value,
            'null_mean': null_values.mean(),
            'null_std': null_values.std(),
            'effect_size': effect_size,
            'statistical_power': power,
            'p_value': (null_values >= actual_value).mean()
        }
    
    return power_analysis

def generate_plots(validation_results, out_dir):
    """Generate comprehensive visualization plots"""
    
    # 1. Cross-validation results across TFs
    plt.figure(figsize=(12, 8))
    tf_metrics = []
    for tf, results in validation_results['multiple_tfs'].items():
        tf_metrics.append({
            'TF': tf,
            'Precision': results['mean_precision'],
            'Recall': results['mean_recall'],
            'F1': results['mean_f1'],
            'n_sites': results['n_sites']
        })
    
    tf_df = pd.DataFrame(tf_metrics)
    metrics = ['Precision', 'Recall', 'F1']
    x = np.arange(len(tf_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, tf_df[metric], width, label=metric)
    
    plt.xlabel('Transcription Factor')
    plt.ylabel('Score')
    plt.title('Performance Across Multiple TFs (5-fold CV)')
    plt.xticks(x + width, tf_df['TF'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'multiple_tf_performance.png'), dpi=300)
    plt.close()
    
    # 2. Feature ablation study
    ablation_df = validation_results['feature_ablation']
    plt.figure(figsize=(10, 6))
    ablation_df.T[['precision', 'recall', 'f1_score']].plot(kind='bar')
    plt.title('Feature Ablation Study')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_ablation.png'), dpi=300)
    plt.close()
    
    # 3. Tool comparison
    tool_df = validation_results['tool_comparison']
    plt.figure(figsize=(10, 6))
    tool_df.T[['precision', 'recall', 'f1_score']].plot(kind='bar')
    plt.title('Comparison with Other Tools')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tool_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive validation for D-Sites TFBS predictor")
    parser.add_argument('--known', required=True, help='Known binding sites TSV')
    parser.add_argument('--fasta', required=True, help='Genome FASTA file')
    parser.add_argument('--gff', required=True, help='Annotation GFF file')
    parser.add_argument('--motif_dir', required=True, help='Directory containing motif files')
    parser.add_argument('--tfs', required=True, help='Comma-separated list of TFs to test')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n_permutations', type=int, default=1000, help='Number of permutations for power analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Parse TFs
    tf_list = args.tfs.split(',')
    
    # Load known sites
    known_df = load_known_sites(args.known)
    
    # Run comprehensive validation
    validation_results = {}
    
    print("1. Running multiple TF evaluation with cross-validation...")
    validation_results['multiple_tfs'] = evaluate_multiple_tfs(
        tf_list, args.known, args.fasta, args.motif_dir, args.gff
    )
    
    print("\n2. Running feature ablation study...")
    # Use first TF for ablation study
    first_tf = tf_list[0]
    tf_known_df = load_known_sites(args.known, first_tf)
    motif_path = os.path.join(args.motif_dir, f"{first_tf}.meme")
    validation_results['feature_ablation'] = feature_ablation_study(
        tf_known_df, args.fasta, motif_path, args.gff
    )
    
    print("\n3. Running tool comparison...")
    validation_results['tool_comparison'] = compare_with_additional_tools(
        tf_known_df, args.fasta, motif_path, args.gff
    )
    
    print("\n4. Running statistical power analysis...")
    validation_results['power_analysis'] = statistical_power_analysis(
        tf_known_df, args.fasta, motif_path, args.gff, args.n_permutations
    )
    
    print("\n5. Generating plots...")
    generate_plots(validation_results, args.out_dir)
    
    # Save results
    results_path = os.path.join(args.out_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for key, value in validation_results.items():
            if hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nValidation complete! Results saved to {args.out_dir}")
    print("Key findings:")
    
    # Print summary
    avg_f1 = np.mean([results['mean_f1'] for results in validation_results['multiple_tfs'].values()])
    print(f"Average F1 score across {len(tf_list)} TFs: {avg_f1:.3f}")
    
    best_config = validation_results['feature_ablation']['f1_score'].idxmax()
    print(f"Best performing configuration: {best_config}")

if __name__ == "__main__":
    main()