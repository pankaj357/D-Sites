#!/usr/bin/env python3
"""
Generate PR curves from D-Sites and FIMO prediction files for TFs from different genomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import argparse
import os

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def load_known_sites(known_data_file, tf_name, genome_accession):
    """
    Load known binding sites from the TSV file for specific TF and genome
    This function is modified to match the benchmarking script's approach
    """
    try:
        # Read the known data file (matching benchmarking script approach)
        known_df = pd.read_csv(known_data_file, sep="\t", dtype=str, comment='#')
        
        # Filter for the specific TF and genome (matching benchmarking script)
        known_df = known_df[(known_df['TF'] == tf_name) & 
                           (known_df['genome_accession'] == genome_accession)].copy()
        
        # Convert numeric columns (matching benchmarking script)
        for col in ['site_start', 'site_end']:
            known_df[col] = pd.to_numeric(known_df[col], errors='coerce')
        
        known_df = known_df.dropna(subset=['site_start', 'site_end'])
        
        print(f"   Found {len(known_df)} known sites for {tf_name} in {genome_accession}")
        
        # Convert to list of dictionaries (maintaining original format)
        known_sites = []
        for _, row in known_df.iterrows():
            # Handle strand notation (-1/1 or +/-)
            strand = row['site_strand']
            if strand == -1 or strand == '-1' or strand == '-':
                strand_str = '-'
            else:
                strand_str = '+'
            
            known_sites.append({
                'contig': row['genome_accession'],
                'start': int(row['site_start']),
                'end': int(row['site_end']),
                'strand': strand_str,
                'sequence': row.get('sequence', ''),
                'tf': row['TF'],
                'evidence': row.get('experimental_evidence', ''),
                'genome': row['genome_accession']
            })
            
        return known_sites
        
    except Exception as e:
        print(f"   Error loading known sites: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_fimo_predictions(fimo_file, genome_accession):
    """
    Load FIMO predictions from TSV file (FIMO default output format)
    """
    try:
        # FIMO files are tab-separated and may have comment lines starting with #
        fimo_df = pd.read_csv(fimo_file, sep='\t', comment='#')
        
        # Check if the file is empty
        if fimo_df.empty:
            print(f"   ⚠️  FIMO file is empty")
            return pd.DataFrame()
        
        # Filter for the specific genome
        fimo_df = fimo_df[fimo_df['sequence_name'] == genome_accession]
        
        # Convert to standard format
        predictions = []
        for _, row in fimo_df.iterrows():
            predictions.append({
                'contig': row['sequence_name'],
                'start': int(row['start']),
                'end': int(row['stop']),
                'strand': row['strand'],
                'score': float(row['score']),
                'p_value': float(row['p-value']),
                'q_value': float(row['q-value']),
                'seq': row['matched_sequence']
            })
        
        print(f"   Loaded {len(predictions)} FIMO predictions for {genome_accession}")
        return pd.DataFrame(predictions)
        
    except Exception as e:
        print(f"   Error loading FIMO predictions: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def match_predictions_with_truth(predictions_df, known_sites, tolerance=10):
    """
    Match predictions with known sites to create true labels
    """
    y_scores = []
    y_true = []
    matched_details = []
    
    for _, pred_row in predictions_df.iterrows():
        score = pred_row['score']
        y_scores.append(score)
        
        # Check if this prediction matches any known site
        is_true = 0
        match_info = None
        
        for known_site in known_sites:
            # Check same contig/genome
            if pred_row['contig'] != known_site['contig']:
                continue
            
            # Check same strand (if available in prediction)
            if 'strand' in pred_row and known_site['strand'] != pred_row['strand']:
                continue
            
            # Check for position overlap with tolerance
            pred_start, pred_end = pred_row['start'], pred_row['end']
            known_start, known_end = known_site['start'], known_site['end']
            
            # Calculate overlap
            overlap_start = max(pred_start, known_start)
            overlap_end = min(pred_end, known_end)
            
            if overlap_start <= overlap_end + tolerance:
                is_true = 1
                match_info = {
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                    'known_start': known_start,
                    'known_end': known_end,
                    'known_sequence': known_site['sequence'],
                    'pred_sequence': pred_row.get('seq', ''),
                    'score': score,
                    'strand': known_site['strand'],
                    'genome': known_site['genome'],
                    'evidence': known_site['evidence']
                }
                break
        
        y_true.append(is_true)
        if match_info:
            matched_details.append(match_info)
    
    return np.array(y_scores), np.array(y_true), matched_details

def plot_pr_curves(tf_data, output_path):
    """
    Plot beautiful PR curves for multiple TFs and tools with distinct colors for each curve
    """
    # Define a color palette with 6 distinct colors
    colors = [
        '#1f77b4',  # blue - AmrZ D-Sites
        '#ff7f0e',  # orange - AmrZ FIMO
        '#2ca02c',  # green - GlxR D-Sites
        '#d62728',  # red - GlxR FIMO
        '#9467bd',  # purple - CodY D-Sites
        '#8c564b',  # brown - CodY FIMO
        '#e377c2',  # pink - Other D-Sites
        '#7f7f7f',  # gray - Other FIMO
        '#bcbd22',  # olive
        '#17becf'   # cyan
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    color_index = 0
    line_styles = {'D-Sites': '-', 'FIMO': '--'}
    
    # Plot each TF's PR curve for each tool
    for tf_name, tools_data in tf_data.items():
        for tool_name, data in tools_data.items():
            if data is not None:  # Only plot if we have data
                y_true, y_scores, _ = data
                precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision)
                ap_score = average_precision_score(y_true, y_scores)
                
                # Get color and line style
                color = colors[color_index % len(colors)]
                linestyle = line_styles.get(tool_name, '-')
                
                label = f'{tf_name} ({tool_name}) - AP = {ap_score:.3f}'
                ax.plot(recall, precision, 
                        color=color, 
                        linestyle=linestyle,
                        linewidth=3, 
                        label=label)
                
                color_index += 1
    
    # Add no-skill line
    if tf_data:
        # Calculate baseline (fraction of positive samples)
        first_tf_data = list(tf_data.values())[0]
        first_tool_data = list(first_tf_data.values())[0]
        if first_tool_data is not None:
            y_true_first = first_tool_data[0]
            baseline = np.mean(y_true_first)
            ax.axhline(y=baseline, color='gray', linestyle=':', 
                      linewidth=2, label=f'No Skill (AP = {baseline:.3f})')
    
    # Customize the plot
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curves - D-Sites vs FIMO TFBS Prediction', fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add some styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"PR curve saved to {output_path}")

def save_detailed_results(tf_data, output_dir):
    """Save detailed matching results for each TF and tool"""
    os.makedirs(output_dir, exist_ok=True)
    
    for tf_name, tools_data in tf_data.items():
        for tool_name, data in tools_data.items():
            if data is not None:
                y_true, y_scores, matched_details = data
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                
                # Save overall statistics
                stats = {
                    'tf_name': tf_name,
                    'tool': tool_name,
                    'total_predictions': len(y_scores),
                    'true_positives': sum(y_true),
                    'false_positives': len(y_scores) - sum(y_true),
                    'precision': sum(y_true) / len(y_scores) if len(y_scores) > 0 else 0,
                    'average_precision': average_precision_score(y_true, y_scores),
                    'auc_pr': auc(recall, precision)
                }
                
                stats_df = pd.DataFrame([stats])
                stats_path = os.path.join(output_dir, f'{tf_name.replace(" ", "_")}_{tool_name}_statistics.csv')
                stats_df.to_csv(stats_path, index=False)
                
                # Save matched details if available
                if matched_details:
                    matched_df = pd.DataFrame(matched_details)
                    matched_path = os.path.join(output_dir, f'{tf_name.replace(" ", "_")}_{tool_name}_matched_sites.csv')
                    matched_df.to_csv(matched_path, index=False)
                    print(f"   Saved {len(matched_details)} matched sites to {matched_path}")
                
                print(f"   Saved statistics to {stats_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate PR curves from D-Sites and FIMO predictions for TFs from different genomes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--amrz', help='AmrZ prediction file and genome (format: d_sites_file:genome_accession:fimo_file)')
    parser.add_argument('--glxr', help='GlxR prediction file and genome (format: d_sites_file:genome_accession:fimo_file)')
    parser.add_argument('--cody', help='Cody prediction file and genome (format: d_sites_file:genome_accession:fimo_file)')
    parser.add_argument('--other', action='append', help='Other TF prediction file (format: tf_name:d_sites_file:genome_accession:fimo_file)')
    parser.add_argument('--known_data', required=True,
                       help='TSV file with known binding sites data')
    parser.add_argument('--output', default='figure1_pr_curves.png',
                       help='Output image path')
    parser.add_argument('--results_dir', default='pr_curve_results',
                       help='Directory for detailed results')
    parser.add_argument('--tolerance', type=int, default=10,
                       help='Position tolerance for matching (bp)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("D-Sites vs FIMO PR Curve Generator (Multi-Genome)")
    print("=" * 60)
    
    tf_data = {}
    
    # Process each TF
    tf_configs = []
    
    if args.amrz:
        parts = args.amrz.split(':')
        if len(parts) == 3:
            tf_configs.append(('AmrZ', parts[0], parts[1], parts[2]))  # TF, d_sites_file, genome, fimo_file
        else:
            print("❌ AmrZ format should be: d_sites_file:genome_accession:fimo_file")
    
    if args.glxr:
        parts = args.glxr.split(':')
        if len(parts) == 3:
            tf_configs.append(('GlxR', parts[0], parts[1], parts[2]))
        else:
            print("❌ GlxR format should be: d_sites_file:genome_accession:fimo_file")
    
    if args.cody:
        parts = args.cody.split(':')
        if len(parts) == 3:
            tf_configs.append(('CodY', parts[0], parts[1], parts[2]))
        else:
            print("❌ Cody format should be: d_sites_file:genome_accession:fimo_file")
    
    if args.other:
        for other_arg in args.other:
            parts = other_arg.split(':')
            if len(parts) == 4:
                tf_configs.append((parts[0], parts[1], parts[2], parts[3]))  # TF, d_sites_file, genome, fimo_file
            else:
                print(f"❌ Other TF format should be: tf_name:d_sites_file:genome_accession:fimo_file")
    
    if not tf_configs:
        print("❌ No valid TF configurations provided")
        print("Use --amrz, --glxr, --cody, or --other arguments")
        return
    
    # Process each TF configuration
    for tf_name, d_sites_file, genome, fimo_file in tf_configs:
        try:
            print(f"\n📊 Processing {tf_name} ({genome})...")
            
            tf_data[tf_name] = {'D-Sites': None, 'FIMO': None}
            
            # Load known sites for this TF and genome
            known_sites = load_known_sites(args.known_data, tf_name, genome)
            
            if not known_sites:
                print(f"   ⚠️  No known sites found for {tf_name} in {genome}")
                continue
            
            # Process D-Sites predictions
            print(f"   D-Sites file: {d_sites_file}")
            try:
                pred_df = pd.read_csv(d_sites_file)
                print(f"   Loaded {len(pred_df)} D-Sites predictions")
                
                y_scores, y_true, matched_details = match_predictions_with_truth(
                    pred_df, known_sites, args.tolerance
                )
                
                print(f"   ✅ {len(y_scores)} total D-Sites predictions")
                print(f"   ✅ {sum(y_true)} true positives ({sum(y_true)/len(y_true)*100:.1f}%)")
                
                if sum(y_true) > 0:
                    ap_score = average_precision_score(y_true, y_scores)
                    print(f"   📈 D-Sites Average Precision: {ap_score:.3f}")
                    tf_data[tf_name]['D-Sites'] = (y_true, y_scores, matched_details)
                else:
                    print(f"   ⚠️  No true positives found for D-Sites")
                    
            except Exception as e:
                print(f"   ❌ Error processing D-Sites file: {e}")
                import traceback
                traceback.print_exc()
            
            # Process FIMO predictions
            print(f"   FIMO file: {fimo_file}")
            try:
                fimo_df = load_fimo_predictions(fimo_file, genome)
                if len(fimo_df) > 0:
                    y_scores, y_true, matched_details = match_predictions_with_truth(
                        fimo_df, known_sites, args.tolerance
                    )
                    
                    print(f"   ✅ {len(y_scores)} total FIMO predictions")
                    print(f"   ✅ {sum(y_true)} true positives ({sum(y_true)/len(y_true)*100:.1f}%)")
                    
                    if sum(y_true) > 0:
                        ap_score = average_precision_score(y_true, y_scores)
                        print(f"   📈 FIMO Average Precision: {ap_score:.3f}")
                        tf_data[tf_name]['FIMO'] = (y_true, y_scores, matched_details)
                    else:
                        print(f"   ⚠️  No true positives found for FIMO")
                else:
                    print(f"   ⚠️  No FIMO predictions found for genome {genome}")
                    
            except Exception as e:
                print(f"   ❌ Error processing FIMO file: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"   ❌ Error processing {tf_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Remove TFs with no data
    tf_data = {tf: tools for tf, tools in tf_data.items() if any(data is not None for data in tools.values())}
    
    if not tf_data:
        print("\n❌ No valid data found for PR curves")
        return
    
    # Generate the plot
    print(f"\n🎨 Generating PR curves...")
    plot_pr_curves(tf_data, args.output)
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    save_detailed_results(tf_data, args.results_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tf_name, tools_data in tf_data.items():
        print(f"{tf_name}:")
        for tool_name, data in tools_data.items():
            if data is not None:
                y_true, y_scores, matched = data
                n_pos = sum(y_true)
                n_total = len(y_true)
                ap_score = average_precision_score(y_true, y_scores)
                print(f"  {tool_name}:")
                print(f"    Predictions: {n_total}, True positives: {n_pos} ({n_pos/n_total*100:.1f}%)")
                print(f"    Average Precision: {ap_score:.3f}")
                print(f"    Matched known sites: {len(matched)}")
        print()
    
    print(f"\n✅ Results saved to:")
    print(f"   PR curve: {args.output}")
    print(f"   Detailed results: {args.results_dir}/")

if __name__ == "__main__":
    main()