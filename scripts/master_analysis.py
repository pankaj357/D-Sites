#!/usr/bin/env python3
"""
master_analysis.py - Fixed typo version
"""

import os
import argparse
import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from Bio import SeqIO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import random
from tqdm import trange

# ----------------------- Helper functions ----------------------- #
def load_preds(path):
    """Load predictions with correct column names"""
    df = pd.read_csv(path, sep='\t' if path.endswith('.tsv') else ',')
    
    # Handle different file types
    if 'fimo' in path.lower():
        df = df.rename(columns={
            'sequence_name': 'contig',
            'start': 'start',
            'stop': 'end',
            'score': 'score'
        })
    else:
        df = df.rename(columns={
            'contig': 'contig',
            'start': 'start', 
            'end': 'end',
            'score': 'score'
        })
    
    for c in ('start', 'end'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['start', 'end'])
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    
    return df

def load_known_sites(path):
    """Load known TFBS sites"""
    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={
        'genome_accession': 'contig',
        'site_start': 'start',
        'site_end': 'end'
    })
    
    for c in ('start', 'end'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['start', 'end'])
    df['start'] = df['start'].astype(int)  # Fixed: astype instead of ast
    df['end'] = df['end'].astype(int)
    
    return df

def load_genes_from_gff(gff_path, contig):
    """Load genes from GFF"""
    genes = []
    with open(gff_path) as fh:
        for ln in fh:
            if ln.startswith('#'): continue
            parts = ln.rstrip('\n').split('\t')
            if len(parts) < 9: continue
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts[:9]
            if seqid != contig: continue
            if ftype.lower() not in ('gene','cds','mrna','transcript'): continue
            try:
                s = int(start); e = int(end)
            except:
                continue
            genes.append({'contig': seqid, 'start': s, 'end': e, 'strand': strand})
    return pd.DataFrame(genes)

def build_tss_promoters_vectorized(genes_df, up=300, down=50):
    """Vectorized promoter building"""
    starts = genes_df['start'].values
    ends = genes_df['end'].values
    strands = genes_df['strand'].values
    
    # Vectorized TSS calculation
    tss_positions = np.where(strands == '+', starts, ends)
    
    # Vectorized promoter region calculation
    promoter_starts = np.where(strands == '+', 
                              np.maximum(1, tss_positions - up),
                              np.maximum(1, tss_positions - down))
    promoter_ends = np.where(strands == '+',
                            tss_positions + down,
                            tss_positions + up)
    
    return np.column_stack([promoter_starts, promoter_ends])

def count_overlaps_vectorized(pred_starts, pred_ends, promoter_starts, promoter_ends):
    """Vectorized overlap counting"""
    # Create broadcasted comparisons
    pred_starts_2d = pred_starts[:, np.newaxis]
    pred_ends_2d = pred_ends[:, np.newaxis]
    promoter_starts_2d = promoter_starts[np.newaxis, :]
    promoter_ends_2d = promoter_ends[np.newaxis, :]
    
    # Check for overlaps: not (pred_end < promoter_start OR pred_start > promoter_end)
    overlaps = ~((pred_ends_2d < promoter_starts_2d) | (pred_starts_2d > promoter_ends_2d))
    
    # Count predictions that overlap with any promoter
    return np.any(overlaps, axis=1).sum()

def genome_length_from_fasta(fasta_path, contig):
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id == contig or rec.name == contig:
            return len(rec.seq)
    raise SystemExit(f"Contig {contig} not found in FASTA")

def build_fasta_mask(fasta_path, contig):
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id == contig or rec.name == contig:
            seq = str(rec.seq).upper()
            arr = np.array(list(seq))
            return ~np.isin(arr, list("ACGT"))
    return None

def sample_random_predictions_vectorized(n_samples, motif_len, contig_len, excluded_mask=None):
    """Vectorized random sampling"""
    if excluded_mask is None:
        excluded_mask = np.zeros(contig_len, dtype=bool)
    
    max_start = contig_len - motif_len + 1
    if max_start < 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Generate random starts
    starts = np.random.randint(1, max_start + 1, n_samples)
    ends = starts + motif_len - 1
    
    # Check for invalid regions (containing non-ACGT)
    valid_mask = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        if excluded_mask[starts[i]-1:ends[i]].any():
            valid_mask[i] = False
    
    # If any invalid, resample those positions
    invalid_indices = np.where(~valid_mask)[0]
    for idx in invalid_indices:
        attempts = 0
        while attempts < 100:
            new_start = np.random.randint(1, max_start + 1)
            new_end = new_start + motif_len - 1
            if not excluded_mask[new_start-1:new_end].any():
                starts[idx] = new_start
                ends[idx] = new_end
                break
            attempts += 1
    
    return starts, ends

def compute_distance_to_tss_vectorized(pred_starts, pred_ends, tss_positions):
    """Fixed vectorized distance calculation"""
    pred_centers = (pred_starts + pred_ends) // 2
    tss_sorted = np.sort(tss_positions)
    
    # Find closest TSS using binary search
    indices = np.searchsorted(tss_sorted, pred_centers)
    
    # Calculate distances to left and right TSS
    dist_right = np.abs(pred_centers - np.take(tss_sorted, indices, mode='clip'))
    dist_left = np.abs(pred_centers - np.take(tss_sorted, indices - 1, mode='clip'))
    
    # Handle edge cases - use float arrays instead of integer
    dist_left = dist_left.astype(float)
    dist_right = dist_right.astype(float)
    
    dist_left[indices == 0] = np.inf
    dist_right[indices == len(tss_sorted)] = np.inf
    
    return np.minimum(dist_left, dist_right)

def fast_permutation_test_vectorized(pred_starts, pred_ends, promoter_intervals, contig_len, motif_len, n_perm=1000, fasta_mask=None):
    """Ultra-fast vectorized permutation test"""
    n_preds = len(pred_starts)
    if n_preds == 0:
        return {'observed': 0, 'mean_random': 0, 'p_emp': 1, 'fold': 0}
    
    promoter_starts = promoter_intervals[:, 0]
    promoter_ends = promoter_intervals[:, 1]
    
    # Count observed overlaps
    observed = count_overlaps_vectorized(pred_starts, pred_ends, promoter_starts, promoter_ends)
    
    # Precompute for speed
    random_counts = np.zeros(n_perm, dtype=int)
    
    if fasta_mask is not None:
        excluded_mask = fasta_mask.astype(bool)
    else:
        excluded_mask = np.zeros(contig_len, dtype=bool)
    
    # Run permutations
    for i in trange(n_perm, desc="Permuting"):
        rand_starts, rand_ends = sample_random_predictions_vectorized(n_preds, motif_len, contig_len, excluded_mask)
        random_counts[i] = count_overlaps_vectorized(rand_starts, rand_ends, promoter_starts, promoter_ends)
    
    mean_random = float(random_counts.mean())
    p_emp = float((random_counts >= observed).sum() + 1) / (n_perm + 1)
    fold = (observed / mean_random) if mean_random > 0 else float('inf')
    
    return {
        'observed': observed,
        'mean_random': mean_random,
        'p_emp': p_emp,
        'fold': fold
    }

def compute_pr_auc_fast(preds_df, known_df, contig):
    """Fast PR curve computation"""
    preds = preds_df[preds_df['contig'] == contig].copy()
    if len(preds) == 0:
        return None, None
    
    # Create interval tree for known sites
    known_tree = IntervalTree()
    for _, row in known_df[known_df['contig'] == contig].iterrows():
        known_tree.addi(row['start'], row['end'] + 1)
    
    # Check overlaps
    y_true = []
    for _, row in preds.iterrows():
        y_true.append(int(known_tree.overlaps(row['start'], row['end'] + 1)))
    
    y_score = preds['score'].values if 'score' in preds.columns else np.ones(len(preds))
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return precision, recall

# ----------------------- Main ----------------------- #
def main():
    parser = argparse.ArgumentParser(description="Ultra-fast TFBS analysis")
    parser.add_argument('--known', required=True)
    parser.add_argument('--tool_preds', required=True)
    parser.add_argument('--fimo', required=True)
    parser.add_argument('--gff', required=True)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--contig', required=True)
    parser.add_argument('--motif_len', type=int, required=True)
    parser.add_argument('--tf', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_permutations', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    known_df = load_known_sites(args.known)
    tool_df = load_preds(args.tool_preds)
    fimo_df = load_preds(args.fimo)
    genes_df = load_genes_from_gff(args.gff, args.contig)
    contig_len = genome_length_from_fasta(args.fasta, args.contig)
    fasta_mask = build_fasta_mask(args.fasta, args.contig)

    # Build promoters
    promoter_intervals = build_tss_promoters_vectorized(genes_df)
    
    # Extract predictions
    tool_contig_preds = tool_df[tool_df['contig'] == args.contig]
    pred_starts = tool_contig_preds['start'].values
    pred_ends = tool_contig_preds['end'].values

    print("Computing PR curves...")
    tool_prec, tool_rec = compute_pr_auc_fast(tool_df, known_df, args.contig)
    fimo_prec, fimo_rec = compute_pr_auc_fast(fimo_df, known_df, args.contig)

    print("Running ultra-fast permutation test...")
    perm = fast_permutation_test_vectorized(
        pred_starts, pred_ends, promoter_intervals, 
        contig_len, args.motif_len, n_perm=args.n_permutations, 
        fasta_mask=fasta_mask
    )

    print("Computing distances to TSS...")
    tss_positions = []
    for _, row in genes_df.iterrows():
        tss_positions.append(row['start'] if row['strand'] == '+' else row['end'])
    tss_positions = np.array(tss_positions)
    
    if len(pred_starts) > 0:
        dists = compute_distance_to_tss_vectorized(pred_starts, pred_ends, tss_positions)
    else:
        dists = np.array([])

    print("Generating plots...")
    
    # Plot 1: PR Curve
    plt.figure(figsize=(8, 6))
    if tool_prec is not None:
        plt.plot(tool_rec, tool_prec, label=f'{args.tf} Top5%', color='tab:blue')
        plt.plot(fimo_rec, fimo_prec, label='FIMO', color='tab:orange')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('Precision-Recall Curve')
    else:
        plt.text(0.5, 0.5, "No predictions", ha='center', va='center')
        plt.title('Precision-Recall Curve (No Data)')
    plt.tight_layout()
    pr_curve_path = os.path.join(args.out_dir, f"{args.tf}_pr_curve.png")
    plt.savefig(pr_curve_path, dpi=600)
    plt.close()

    # Plot 2: Overlap Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(['Observed', 'Expected'], [perm['observed'], perm['mean_random']],
            color=['tab:blue', 'tab:orange'])
    plt.ylabel('Overlapping promoters')
    plt.title('Promoter Overlap Comparison')
    plt.tight_layout()
    overlap_path = os.path.join(args.out_dir, f"{args.tf}_overlap_comparison.png")
    plt.savefig(overlap_path, dpi=600)
    plt.close()

    # Plot 3: Distance Histogram
    plt.figure(figsize=(8, 6))
    if len(dists) > 0:
        # Filter out infinite distances for plotting
        finite_dists = dists[np.isfinite(dists)]
        if len(finite_dists) > 0:
            plt.hist(finite_dists, bins=50, color='tab:green')
            plt.xlabel('Distance to TSS (bp)')
            plt.ylabel('Frequency')
            plt.title('Distance to Nearest TSS')
        else:
            plt.text(0.5, 0.5, "No finite distances", ha='center', va='center')
            plt.title('Distance to Nearest TSS (No Finite Distances)')
    else:
        plt.text(0.5, 0.5, "No predictions", ha='center', va='center')
        plt.title('Distance to Nearest TSS (No Data)')
    plt.tight_layout()
    distance_path = os.path.join(args.out_dir, f"{args.tf}_distance_histogram.png")
    plt.savefig(distance_path, dpi=600)
    plt.close()

    # Save results
    summary = {
        'n_predictions': len(tool_contig_preds),
        'observed_overlaps': perm['observed'],
        'expected_overlaps': perm['mean_random'],
        'fold_enrichment': perm['fold'],
        'p_value': perm['p_emp']
    }
    outcsv = os.path.join(args.out_dir, f"{args.tf}_summary.csv")
    pd.Series(summary).to_csv(outcsv)

    print(f"[DONE] Results saved to {args.out_dir}")
    print(f"Observed: {perm['observed']}, Expected: {perm['mean_random']:.1f}, Fold: {perm['fold']:.1f}, p-value: {perm['p_emp']:.3e}")

if __name__ == "__main__":
    main()