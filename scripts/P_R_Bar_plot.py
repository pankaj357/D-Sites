# generate_performance_summary.py
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
data_dir = "./benchmark_results"  # Directory containing your JSON files
output_file = "./summary_performance_comparison.png"
tfs_to_include = ["AmrZ", "GlxR", "CodY"]  # The TFs you want to plot

def extract_data_from_json(filepath):
    """Extract relevant metrics from a single JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get TF name and prediction type from filename
    filename = os.path.basename(filepath)
    tf_name = None
    prediction_type = None
    
    for tf in tfs_to_include:
        if tf in filename:
            tf_name = tf
            if "top" in filename.lower() or "top5" in filename:
                prediction_type = "top5"
            else:
                prediction_type = "full"
            break
    
    if not tf_name or not prediction_type:
        return None
    
    return {
        "tf": tf_name,
        "type": prediction_type,
        "fimo_recall": data.get("fimo_recall", 0),
        "dsites_recall": data.get("dsites_recall", 0),
        "fimo_precision": data.get("fimo_precision", 0),
        "dsites_precision": data.get("dsites_precision", 0)
    }

def main():
    # Find all JSON files in the data directory
    json_files = glob.glob(f"{data_dir}/*.json")
    
    # Parse all files
    all_data = {}
    for filepath in json_files:
        result = extract_data_from_json(filepath)
        if result:
            tf = result["tf"]
            pred_type = result["type"]
            
            if tf not in all_data:
                all_data[tf] = {"full": {}, "top5": {}}
            
            all_data[tf][pred_type] = {
                "fimo_recall": result["fimo_recall"],
                "dsites_recall": result["dsites_recall"],
                "fimo_precision": result["fimo_precision"],
                "dsites_precision": result["dsites_precision"]
            }
    
    # Prepare data for plotting
    tf_names = []
    full_recall_data = []  # [fimo_full_recall, dsites_full_recall] for each TF
    top5_precision_data = []  # [fimo_full_precision, dsites_top5_precision] for each TF
    
    for tf in tfs_to_include:
        if tf in all_data:
            tf_names.append(tf)
            
            # Get recall values from FULL predictions
            fimo_full_recall = all_data[tf]["full"].get("fimo_recall", 0)
            dsites_full_recall = all_data[tf]["full"].get("dsites_recall", 0)
            full_recall_data.append([fimo_full_recall, dsites_full_recall])
            
            # Get precision values: FIMO (full) vs D-Sites (top5%)
            fimo_full_precision = all_data[tf]["full"].get("fimo_precision", 0)
            dsites_top5_precision = all_data[tf]["top5"].get("dsites_precision", 0)
            top5_precision_data.append([fimo_full_precision, dsites_top5_precision])
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Set bar width and positions
    x = np.arange(len(tf_names))
    width = 0.35
    
    # Plot Recall comparison (Full predictions)
    bars1_fimo = ax1.bar(x - width/2, [d[0] for d in full_recall_data], width, label='FIMO', color='#FF9999', edgecolor='black')
    bars1_dsites = ax1.bar(x + width/2, [d[1] for d in full_recall_data], width, label='D-Sites', color='#66B2FF', edgecolor='black')
    
    ax1.set_xlabel('Transcription Factor', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax1.set_title('Recall: Full Prediction Sets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tf_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot Precision comparison (FIMO Full vs D-Sites Top 5%)
    bars2_fimo = ax2.bar(x - width/2, [d[0] for d in top5_precision_data], width, label='FIMO (Full)', color='#FF9999', edgecolor='black')
    bars2_dsites = ax2.bar(x + width/2, [d[1] for d in top5_precision_data], width, label='D-Sites (Top 5%)', color='#66B2FF', edgecolor='black')
    
    ax2.set_xlabel('Transcription Factor', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision: FIMO (Full) vs D-Sites (Top 5%)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tf_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels INSIDE the bars for better visibility
    def add_value_labels_inside(ax, bars, data, fontsize=9):
        for bar, value in zip(bars, data):
            height = bar.get_height()
            # Position text in the middle of the bar
            ax.text(bar.get_x() + bar.get_width()/2., height/2, f'{value:.3f}',
                   ha='center', va='center', fontweight='bold', fontsize=fontsize,
                   color='white' if height > 0.05 else 'black')  # White text on dark bars
    
    # Add labels to recall plot
    add_value_labels_inside(ax1, bars1_fimo, [d[0] for d in full_recall_data], fontsize=10)
    add_value_labels_inside(ax1, bars1_dsites, [d[1] for d in full_recall_data], fontsize=10)
    
    # Add labels to precision plot
    add_value_labels_inside(ax2, bars2_fimo, [d[0] for d in top5_precision_data], fontsize=10)
    add_value_labels_inside(ax2, bars2_dsites, [d[1] for d in top5_precision_data], fontsize=10)
    
    # Set appropriate y-axis limits
    ax1.set_ylim(0, max([max(vals) for vals in full_recall_data]) * 1.15)
    ax2.set_ylim(0, max([max(vals) for vals in top5_precision_data]) * 1.5)  # More space for precision
    
    # Add improvement annotations for precision plot
    for i, (fimo_prec, dsites_prec) in enumerate(top5_precision_data):
        improvement = (dsites_prec - fimo_prec) / fimo_prec
        ax2.text(x[i], max(fimo_prec, dsites_prec) * 1.1, f'{improvement:+.0%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure saved as {output_file}")
    
    # Print summary
    print("\nPerformance Summary:")
    for i, tf in enumerate(tf_names):
        fimo_recall, dsites_recall = full_recall_data[i]
        fimo_precision, dsites_precision = top5_precision_data[i]
        recall_improvement = (dsites_recall - fimo_recall) / fimo_recall * 100 if fimo_recall > 0 else 0
        precision_improvement = (dsites_precision - fimo_precision) / fimo_precision * 100 if fimo_precision > 0 else 0
        
        print(f"\n{tf}:")
        print(f"  Recall: FIMO={fimo_recall:.3f}, D-Sites={dsites_recall:.3f} ({recall_improvement:+.1f}%)")
        print(f"  Precision: FIMO={fimo_precision:.3f}, D-Sites(Top5%)={dsites_precision:.3f} ({precision_improvement:+.1f}%)")

if __name__ == "__main__":
    main()