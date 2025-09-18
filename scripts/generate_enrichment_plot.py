# generate_enrichment_figure.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Configuration
data_dir = "./enrichment_fig"  # Directory containing your CSV files
output_file = "./promoter_enrichment.png"
tfs_to_include = ["AmrZ", "GlxR", "CodY"]  # The TFs you want to plot

def extract_enrichment_data(filepath):
    """Extract enrichment metrics from a single CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(filepath, index_col=0)
        data = df.iloc[:, 0].to_dict()
        
        # Get TF name from filename
        filename = os.path.basename(filepath)
        tf_name = None
        
        for tf in tfs_to_include:
            if tf in filename:
                tf_name = tf
                break
        
        if not tf_name:
            return None
        
        return {
            "tf": tf_name,
            "n_predictions": data.get("n_predictions", 0),
            "observed": data.get("observed_overlaps", 0),
            "expected": data.get("expected_overlaps", 0),
            "fold_enrichment": data.get("fold_enrichment", 0),
            "p_value": data.get("p_value", 1.0)
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    # Find all CSV files in the data directory
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Parse all files and extract enrichment data
    enrichment_data = {}
    for filepath in csv_files:
        result = extract_enrichment_data(filepath)
        if result:
            tf = result["tf"]
            enrichment_data[tf] = result
    
    if not enrichment_data:
        print("No valid enrichment data found in the CSV files")
        return
    
    print("Enrichment data extracted from CSV files:")
    for tf, data in enrichment_data.items():
        print(f"{tf}: Fold enrichment={data['fold_enrichment']:.2f}, p={data['p_value']:.2e}")
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data for plotting
    tf_names = list(enrichment_data.keys())
    fold_enrichment = [enrichment_data[tf]["fold_enrichment"] for tf in tf_names]
    p_values = [enrichment_data[tf]["p_value"] for tf in tf_names]
    observed = [enrichment_data[tf]["observed"] for tf in tf_names]
    expected = [enrichment_data[tf]["expected"] for tf in tf_names]
    
    # Plot 1: Fold Enrichment Bar Chart
    bars = ax1.bar(tf_names, fold_enrichment, color=['#4C72B0', '#DD8452', '#55A868'], 
                   edgecolor='black', alpha=0.8)
    
    ax1.set_xlabel('Transcription Factor', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fold Enrichment', fontsize=12, fontweight='bold')
    ax1.set_title('Promoter Region Enrichment of Top 5% Predictions', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels INSIDE bars
    for bar, value in zip(bars, fold_enrichment):
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() - 0.1,   # slightly inside the bar
            f'{value:.2f}x',
            ha='center', va='top', color='white',
            fontweight='bold', fontsize=11
        )
    
    # Add horizontal line at 1 (no enrichment)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(len(tf_names)-0.5, 1.05, 'No enrichment (1.0x)', 
             ha='right', va='bottom', color='red', fontweight='bold')
    
    # Plot 2: Observed vs Expected Overlaps
    x = np.arange(len(tf_names))
    width = 0.35
    
    bars_observed = ax2.bar(x - width/2, observed, width, label='Observed', 
                           color='#2E86AB', edgecolor='black', alpha=0.8)
    bars_expected = ax2.bar(x + width/2, expected, width, label='Expected', 
                           color='#A23B72', edgecolor='black', alpha=0.8)
    
    ax2.set_xlabel('Transcription Factor', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Overlaps', fontsize=12, fontweight='bold')
    ax2.set_title('Observed vs Expected Promoter Overlaps', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tf_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels INSIDE bars
    for bar_obs, bar_exp in zip(bars_observed, bars_expected):
        ax2.text(
            bar_obs.get_x() + bar_obs.get_width()/2, 
            bar_obs.get_height() - 5,   # inside
            f'{int(bar_obs.get_height())}',
            ha='center', va='top', color='white',
            fontweight='bold', fontsize=9
        )
        ax2.text(
            bar_exp.get_x() + bar_exp.get_width()/2, 
            bar_exp.get_height() - 5,
            f'{int(bar_exp.get_height())}',
            ha='center', va='top', color='white',
            fontweight='bold', fontsize=9
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed summary
    print("\n" + "="*60)
    print("Promoter Enrichment Summary:")
    print("="*60)
    for tf, data in enrichment_data.items():
        print(f"\n{tf}:")
        print(f"  Predictions analyzed: {data['n_predictions']:.0f}")
        print(f"  Observed overlaps: {data['observed']:.0f}")
        print(f"  Expected overlaps: {data['expected']:.1f}")
        print(f"  Fold enrichment: {data['fold_enrichment']:.2f}x")
        print(f"  p-value: {data['p_value']:.2e}")

if __name__ == "__main__":
    main()
