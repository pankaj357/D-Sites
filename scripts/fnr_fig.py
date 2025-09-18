# figure4_fnr_validation.py
import matplotlib.pyplot as plt

# Data (excluding Top 1%)
tools = ["FIMO", "D-Sites (Full)", "D-Sites (Top 5%)"]
genes_with_hits = [4, 47, 5]
validation_success = [5.80, 68.12, 7.25]  # percentages
total_predictions = [895, 5394, 269]

# Colors
bar_colors = ["#4C72B0", "#55A868", "#DD8452"]

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Plot bar chart (genes with hits)
bars = ax.bar(tools, genes_with_hits, color=bar_colors, edgecolor="black", alpha=0.85)

# Labels
ax.set_ylabel("Number of FNR-Regulated Genes with Promoter Hits", fontsize=12, fontweight="bold")
ax.set_title("Functional Validation Against the FNR Regulon", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Add values inside bars (genes with hits)
for bar, value in zip(bars, genes_with_hits):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() - 1,  # slightly inside
        f"{value}",
        ha="center", va="top", color="white",
        fontweight="bold", fontsize=11
    )

# Add success rate (%) and total predictions above bars
for bar, success, total in zip(bars, validation_success, total_predictions):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f"{success:.1f}% (Total={total})",
        ha="center", va="bottom", color="black",
        fontsize=10, fontweight="bold"
    )

# Save figure
plt.tight_layout()
plt.savefig("Figure4_FNR_validation.png", dpi=300, bbox_inches="tight")
plt.show()
