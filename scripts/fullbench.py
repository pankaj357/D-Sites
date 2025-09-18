#!/usr/bin/env python3
"""
Comprehensive TFBS Benchmarking Script (updated evaluation)
- Compares D-Sites predictions vs FIMO with full statistical validation
- Includes recall, precision, F1, AUC, false positives, runtime, and significance testing

Improvements in this version (only evaluation logic changed):
- Recall is computed as fraction of unique known sites hit (prevents overcounting).
- AUC (PR/ROC) is computed at site-level: known positives + sampled negatives,
  where each region receives the max prediction score overlapping it (or 0).
- True FPR is computed using negative control regions (in addition to FDR = 1-precision).
- JSON serialization helper included.
"""

import os
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
from intervaltree import Interval, IntervalTree
import time
import psutil
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import json
from statsmodels.stats.contingency_tables import mcnemar
from joblib import Parallel, delayed
import random

def make_json_serializable(obj):
    """Recursively convert numpy and pandas types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return obj

# ---------------------------
# Optimized Sequence Matching
# ---------------------------
def optimized_sequence_matches(known_df, pred_df, genome_seq):
    """Efficient sequence matching without nested loops"""
    known_seqs = set( genome_seq[int(row['site_start'])-1:int(row['site_end'])] 
                      for _, row in known_df.iterrows())
    known_revcomps = set(str(Seq(seq).reverse_complement()) for seq in known_seqs)
    
    pred_seqs = pred_df['seq'].astype(str).values
    
    exact = np.isin(pred_seqs, list(known_seqs)).sum()
    revcomp = np.isin(pred_seqs, list(known_revcomps)).sum()
    
    # Substring matches (exact+revcomp removed)
    remaining = set(pred_seqs) - known_seqs - known_revcomps
    substring = 0
    for seq in remaining:
        if any(seq in kseq or kseq in seq for kseq in known_seqs):
            substring += 1
            
    return int(exact), int(revcomp), int(substring)

# ---------------------------
# Interval-based Overlap Calculation (kept for some utilities)
# ---------------------------
def calculate_overlaps(pred_df, known_tree):
    """Calculate overlaps using interval tree (counts predictions overlapping known sites)."""
    overlaps = 0
    for _, pred in pred_df.iterrows():
        if known_tree.overlaps(pred['start'], pred['end'] + 1):
            overlaps += 1
    return overlaps

# ---------------------------
# Unique-known-site-based Recall & FN (NEW, recommended)
# ---------------------------
def recall_unique_known_hit(pred_df, known_tree):
    """
    Compute recall as fraction of known sites that are hit at least once by any prediction.
    Returns (recall, n_known_hit, total_known).
    """
    if len(known_tree) == 0:
        return 0.0, 0, 0
    # build prediction interval tree
    pred_tree = IntervalTree(Interval(int(r['start']), int(r['end'])+1) for _, r in pred_df.iterrows())
    hit_count = 0
    total_known = 0
    for iv in known_tree:
        total_known += 1
        if pred_tree.overlaps(iv.begin, iv.end):
            hit_count += 1
    recall = hit_count / total_known if total_known > 0 else 0.0
    return recall, hit_count, total_known

def false_negatives_unique(pred_df, known_tree):
    """
    Compute number and rate of false negatives based on unique-known-site hits.
    Returns (fn_count, fn_rate).
    """
    recall, hit_count, total_known = recall_unique_known_hit(pred_df, known_tree)
    fn_count = total_known - hit_count
    fn_rate = fn_count / total_known if total_known > 0 else 0.0
    return fn_count, fn_rate

# ---------------------------
# Precision, F1 (keep previous semantics)
# ---------------------------
def calculate_precision(pred_df, known_tree):
    """Calculate precision as fraction of predictions overlapping known sites."""
    overlaps = calculate_overlaps(pred_df, known_tree)
    return overlaps / len(pred_df) if len(pred_df) > 0 else 0

def calculate_f1_score(precision, recall):
    """Calculate F1 score"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# ---------------------------
# False Positive Analysis (keeps prior semantics for FP_count w.r.t predictions)
# ---------------------------
def calculate_false_positives(pred_df, known_tree):
    """Calculate false positives (count of predictions that do not overlap known sites)."""
    fp_count = 0
    for _, pred in pred_df.iterrows():
        if not known_tree.overlaps(pred['start'], pred['end'] + 1):
            fp_count += 1
    fp_rate = fp_count / len(pred_df) if len(pred_df) > 0 else 0
    return fp_count, fp_rate

# ---------------------------
# AUC Calculation (site-level) - NEW
# ---------------------------
def compute_auc_site_level(pred_df, known_tree, genome_length, n_negative_samples=1000, neg_region_size=None, random_seed=42):
    """
    Compute PR and ROC AUC at site-level.

    Approach:
    - For each known interval (from known_tree), compute the max prediction score overlapping it (or 0).
    - Sample a set of negative regions (not overlapping known sites) of length neg_region_size (or average known length)
      and compute the max prediction score overlapping each negative region (or 0).
    - Use these labeled regions to compute PR and ROC curves and AUC.

    This avoids only scoring predictions and gives a fair comparison of scores.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Build pred tree with scores as data
    pred_tree = IntervalTree()
    for _, r in pred_df.iterrows():
        s = int(r['start']); e = int(r['end'])
        score = float(r.get('score', 0.0))
        pred_tree.addi(s, e+1, score)

    # Get known regions list (start,end)
    known_regions = [(iv.begin, iv.end-1) for iv in known_tree]  # end-1 to get original inclusive end
    if len(known_regions) == 0:
        return 0.0, 0.0

    # Determine negative region size: use median known site length if not provided
    if neg_region_size is None:
        lengths = [ (iv.end - iv.begin) for iv in known_tree ]
        neg_region_size = int(np.median(lengths)) if lengths else 1
        if neg_region_size < 1:
            neg_region_size = 1

    # Sample negative regions without overlapping known_tree
    negative_regions = []
    attempts = 0
    max_attempts = n_negative_samples * 50
    while len(negative_regions) < n_negative_samples and attempts < max_attempts:
        attempts += 1
        start = np.random.randint(0, max(1, genome_length - neg_region_size))
        end = start + neg_region_size
        if not known_tree.overlaps(start, end):
            negative_regions.append((start, end-1))
    # If we sampled none, fallback to 0-length negatives (rare)
    if len(negative_regions) == 0:
        negative_regions = [(0,0)] * min(100, len(known_regions))

    # For each region (positive + negative), compute max prediction score overlapping it
    y_true = []
    y_score = []
    for s,e in known_regions:
        hits = pred_tree.overlap(s, e+1)
        max_score = max((iv.data for iv in hits), default=0.0)
        y_true.append(1)
        y_score.append(max_score)
    for s,e in negative_regions:
        hits = pred_tree.overlap(s, e+1)
        max_score = max((iv.data for iv in hits), default=0.0)
        y_true.append(0)
        y_score.append(max_score)

    y_true = np.array(y_true)
    y_score = np.array(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
    except:
        pr_auc = 0.0
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except:
        roc_auc = 0.0

    return pr_auc, roc_auc

# ---------------------------
# Statistical Significance Testing (kept)
# ---------------------------
def mcnemar_test(tool_df, fimo_df, known_tree):
    """Efficient McNemar using interval tree queries"""
    tool_intervals = IntervalTree(Interval(row['start'], row['end']+1) for _, row in tool_df.iterrows())
    fimo_intervals = IntervalTree(Interval(row['start'], row['end']+1) for _, row in fimo_df.iterrows())
    
    both_correct = only_tool = only_fimo = neither = 0
    
    for interval in known_tree:
        start, end = interval.begin, interval.end
        tool_hit = bool(tool_intervals.overlaps(start, end))
        fimo_hit = bool(fimo_intervals.overlaps(start, end))
        
        if tool_hit and fimo_hit:
            both_correct += 1
        elif tool_hit and not fimo_hit:
            only_tool += 1
        elif not tool_hit and fimo_hit:
            only_fimo += 1
        else:
            neither += 1
            
    table = [[both_correct, only_fimo],
             [only_tool, neither]]
    
    try:
        result = mcnemar(table, exact=True)
        return result.statistic, result.pvalue
    except:
        return None, None

# ---------------------------
# Negative Control Validation (kept)
# ---------------------------
def generate_negative_regions(genome_length, known_tree, n=1000, region_size=1000):
    negative_regions = []

    def attempt_region(_):
        start = np.random.randint(0, genome_length - region_size)
        end = start + region_size
        if not known_tree.overlaps(start, end):
            return (start, end)
        return None

    results = Parallel(n_jobs=-1)(delayed(attempt_region)(i) for i in range(n*10))
    negative_regions = [r for r in results if r][:n]
    return negative_regions

def count_false_positives_in_regions(pred_df, negative_regions):
    """Count FPs in negative regions using interval tree"""
    pred_tree = IntervalTree(Interval(row['start'], row['end']+1) for _, row in pred_df.iterrows())
    fp_count = 0
    for start, end in negative_regions:
        if pred_tree.overlaps(start, end):
            fp_count += 1
    return fp_count

# ---------------------------
# Performance Monitoring (kept)
# ---------------------------
class PerformanceMonitor:
    """Monitor runtime and memory usage"""
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        
    def stop(self):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        return {
            'runtime_seconds': end_time - self.start_time,
            'memory_mb': (end_memory - self.start_memory) / 1024 / 1024
        }

# ---------------------------
# Main Benchmarking Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive TFBS benchmarking tool")
    parser.add_argument("--known", required=True, help="Master collectf_export.tsv")
    parser.add_argument("--tool_preds", required=True, help="D-Sites predictions CSV")
    parser.add_argument("--fimo", required=True, help="FIMO TSV output")
    parser.add_argument("--fasta", required=True, help="Genome FASTA")
    parser.add_argument("--contig", required=True, help="Contig/Genome accession")
    parser.add_argument("--motif_len", type=int, required=True)
    parser.add_argument("--tf", required=True, help="TF name")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_negative_regions", type=int, default=1000, help="Number of negative control regions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Start performance monitoring
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()

    # Load genome sequence
    genome_seq = None
    genome_length = 0
    for rec in SeqIO.parse(args.fasta, "fasta"):
        if rec.id == args.contig:
            genome_seq = str(rec.seq).upper()
            genome_length = len(genome_seq)
            break
    
    if genome_seq is None:
        raise SystemExit(f"Contig {args.contig} not found in FASTA")

    # Load known sites and filter by TF + contig
    known = pd.read_csv(args.known, sep="\t", dtype=str, comment='#')
    known = known[(known['TF'] == args.tf) & (known['genome_accession'] == args.contig)].copy()
    for c in ['site_start','site_end']:
        known[c] = pd.to_numeric(known[c], errors='coerce')
    known = known.dropna(subset=['site_start','site_end'])
    print(f"[INFO] Number of filtered known sites: {known.shape[0]}")

    # Build interval tree for known sites
    known_tree = IntervalTree()
    for _, row in known.iterrows():
        known_tree.addi(row['site_start'], row['site_end'] + 1)

    # Load D-Sites predictions
    dsites_df = pd.read_csv(args.tool_preds, dtype=str)
    for c in ['start','end']:
        dsites_df[c] = pd.to_numeric(dsites_df[c], errors='coerce')
    dsites_df = dsites_df.dropna(subset=['start','end'])
    
    # Handle missing score column
    if 'score' not in dsites_df.columns:
        dsites_df['score'] = 1.0
    else:
        dsites_df['score'] = pd.to_numeric(dsites_df['score'], errors='coerce').fillna(0.0)
    
    print(f"[INFO] Number of predicted sites by D-Sites: {dsites_df.shape[0]}")

    # Load FIMO predictions
    fimo_df = pd.read_csv(args.fimo, sep="\t", comment="#", dtype=str)
    fimo_df = fimo_df[fimo_df['sequence_name'] == args.contig].copy()
    fimo_df['start'] = pd.to_numeric(fimo_df['start'], errors='coerce')
    fimo_df['stop'] = pd.to_numeric(fimo_df['stop'], errors='coerce')
    fimo_df = fimo_df.dropna(subset=['start','stop'])
    
    # Handle FIMO score conversion
    if 'p-value' in fimo_df.columns:
        fimo_df['score'] = -np.log10(pd.to_numeric(fimo_df['p-value'], errors='coerce').fillna(1.0))
    else:
        fimo_df['score'] = 1.0
    
    fimo_df = fimo_df.rename(columns={'stop': 'end'})
    print(f"[INFO] Number of FIMO predictions: {fimo_df.shape[0]}")

    # ---------------------------
    # Evaluation metrics (improved)
    # ---------------------------
    # Recall (unique-known-site hits)
    dsites_recall, dsites_known_hits, total_known = recall_unique_known_hit(dsites_df, known_tree)
    fimo_recall, fimo_known_hits, _ = recall_unique_known_hit(fimo_df, known_tree)

    # Precision (keeps prior definition: fraction of predictions overlapping known)
    dsites_precision = calculate_precision(dsites_df, known_tree)
    fimo_precision = calculate_precision(fimo_df, known_tree)

    # F1 (from precision and unique-known recall)
    dsites_f1 = calculate_f1_score(dsites_precision, dsites_recall)
    fimo_f1 = calculate_f1_score(fimo_precision, fimo_recall)

    # False positives (predictions not overlapping known) and FDR
    dsites_fp_count, dsites_fp_rate = calculate_false_positives(dsites_df, known_tree)
    fimo_fp_count, fimo_fp_rate = calculate_false_positives(fimo_df, known_tree)

    # False negatives (unique-known)
    dsites_fn_count, dsites_fn_rate = false_negatives_unique(dsites_df, known_tree)
    fimo_fn_count, fimo_fn_rate = false_negatives_unique(fimo_df, known_tree)

    # AUCs computed at site-level (positives = known sites; negatives = sampled non-known regions)
    pr_auc_dsites, roc_auc_dsites = compute_auc_site_level(dsites_df, known_tree, genome_length, n_negative_samples=2000, neg_region_size=None)
    pr_auc_fimo, roc_auc_fimo = compute_auc_site_level(fimo_df, known_tree, genome_length, n_negative_samples=2000, neg_region_size=None)

    # Sequence matches
    exact, revcomp, substring = optimized_sequence_matches(known, dsites_df, genome_seq)

    # Statistical significance testing (McNemar)
    mcnemar_stat, mcnemar_pvalue = mcnemar_test(dsites_df, fimo_df, known_tree)

    # Negative control validation (use your provided generator)
    negative_regions = generate_negative_regions(genome_length, known_tree, args.n_negative_regions)
    dsites_fp_neg = count_false_positives_in_regions(dsites_df, negative_regions)
    fimo_fp_neg = count_false_positives_in_regions(fimo_df, negative_regions)

    # True FPR (from negative-region coverage): FP regions / total negative regions
    dsites_fpr_neg = dsites_fp_neg / len(negative_regions) if negative_regions else 0.0
    fimo_fpr_neg = fimo_fp_neg / len(negative_regions) if negative_regions else 0.0

    # Performance metrics
    perf_metrics = perf_monitor.stop()

    # Save comprehensive results
    summary = {
        'known_sites': int(total_known),
        'dsites_predictions': int(len(dsites_df)),
        'fimo_predictions': int(len(fimo_df)),
        
        # Recall metrics (unique-known)
        'dsites_recall': float(dsites_recall),
        'fimo_recall': float(fimo_recall),
        'recall_improvement': float(dsites_recall - fimo_recall),
        'dsites_known_hits': int(dsites_known_hits),
        'fimo_known_hits': int(fimo_known_hits),
        
        # Precision metrics (prediction-level)
        'dsites_precision': float(dsites_precision),
        'fimo_precision': float(fimo_precision),
        'precision_improvement': float(dsites_precision - fimo_precision),
        
        # F1 scores
        'dsites_f1': float(dsites_f1),
        'fimo_f1': float(fimo_f1),
        'f1_improvement': float(dsites_f1 - fimo_f1),
        
        # Error rates (prediction-level false positives and unique-known FNs)
        'dsites_fp_count': int(dsites_fp_count),
        'dsites_fp_rate': float(dsites_fp_rate),   # this is FDR-like (1 - precision) as computed above
        'fimo_fp_count': int(fimo_fp_count),
        'fimo_fp_rate': float(fimo_fp_rate),
        'dsites_fn_count': int(dsites_fn_count),
        'dsites_fn_rate': float(dsites_fn_rate),
        'fimo_fn_count': int(fimo_fn_count),
        'fimo_fn_rate': float(fimo_fn_rate),
        
        # AUC scores (site-level)
        'pr_auc_dsites': float(pr_auc_dsites),
        'pr_auc_fimo': float(pr_auc_fimo),
        'roc_auc_dsites': float(roc_auc_dsites),
        'roc_auc_fimo': float(roc_auc_fimo),
        
        # Sequence matches
        'exact_matches': int(exact),
        'revcomp_matches': int(revcomp),
        'substring_matches': int(substring),
        
        # Statistical significance
        'mcnemar_statistic': mcnemar_stat,
        'mcnemar_pvalue': mcnemar_pvalue,
        'statistically_significant': bool(mcnemar_pvalue < 0.05) if mcnemar_pvalue is not None else None,
        
        # Negative control results (true FPR estimated from negative regions)
        'negative_regions_tested': int(len(negative_regions)),
        'dsites_fp_negative_regions': int(dsites_fp_neg),
        'fimo_fp_negative_regions': int(fimo_fp_neg),
        'dsites_fpr_negative_regions': float(dsites_fpr_neg),
        'fimo_fpr_negative_regions': float(fimo_fpr_neg),
        
        # Performance metrics
        'runtime_seconds': float(perf_metrics['runtime_seconds']),
        'memory_usage_mb': float(perf_metrics['memory_mb'])
    }

    # Save results (convert numpy types to native python for JSON/CSV compatibility)
    out_file = os.path.join(args.out_dir, "comprehensive_benchmark_results.json")
    summary_serializable = make_json_serializable(summary)
    with open(out_file, 'w') as f:
        json.dump(summary_serializable, f, indent=2)

    # Also save as CSV for easy viewing (use serializable summary to avoid numpy types)
    csv_file = os.path.join(args.out_dir, "benchmark_summary.csv")
    pd.Series(summary_serializable).to_csv(csv_file)

    print(f"[DONE] Comprehensive benchmark results saved to {out_file} and {csv_file}")
    print("\n=== KEY RESULTS ===")
    print(f"Recall (unique-known): D-Sites={dsites_recall:.3f}, FIMO={fimo_recall:.3f}, Improvement={dsites_recall-fimo_recall:.3f}")
    print(f"Precision (prediction-level): D-Sites={dsites_precision:.3f}, FIMO={fimo_precision:.3f}, Improvement={dsites_precision-fimo_precision:.3f}")
    print(f"F1 Score: D-Sites={dsites_f1:.3f}, FIMO={fimo_f1:.3f}, Improvement={dsites_f1-fimo_f1:.3f}")
    # print both FDR-like metric and true FPR from negative regions
    print(f"False Positives (prediction-level): D-Sites_count={dsites_fp_count}, D-Sites_FDR={dsites_fp_rate:.3f}")
    print(f"False Positive Rate (negative regions): D-Sites={dsites_fpr_neg:.3f}, FIMO={fimo_fpr_neg:.3f}")
    
    if mcnemar_pvalue is not None:
        print(f"Statistical Significance (McNemar): p={mcnemar_pvalue:.4f} {'(SIGNIFICANT)' if mcnemar_pvalue < 0.05 else '(NOT SIGNIFICANT)'}")

if __name__ == "__main__":
    main()