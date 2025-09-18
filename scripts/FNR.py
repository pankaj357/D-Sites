#!/usr/bin/env python3
"""
Validate TFBS Predictions Against Known Regulated Genes (Exact Gene Match Only)

Steps:
1. Load regulated genes list (CSV/Excel) with column 'Gene'.
2. Map to genome annotation (GFF) using exact gene name match.
3. Extract promoter regions (default -300/+50 bp window).
4. Check if predicted TFBS (CSV) fall within promoters.
5. Compute distance of each hit from TSS.
6. Output detailed matches + unmatched + no-hit genes + summary.
7. Generate publication-quality figures (600 dpi).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_gff(gff_file):
    """Parse GFF and extract gene entries."""
    gff_entries = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
            attr_dict = {}
            for attr in attributes.split(";"):
                if "=" in attr:
                    key, val = attr.split("=", 1)
                    attr_dict[key.strip()] = val.strip()
            gff_entries.append({
                "seqid": seqid,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "gene": attr_dict.get("gene", "").lower()
            })
    return pd.DataFrame(gff_entries)


def make_visualizations(results_df, total_genes, mapped_count, promoter_hit_count, out_prefix):
    """Generate publication-quality visualizations (600 dpi)."""
    if results_df.empty:
        print("No results to visualize.")
        return

    sns.set_style("whitegrid")

    # 1. Coverage bar plot
    coverage_data = {
        "Category": ["Total regulated", "Mapped to GFF", "With promoter hits"],
        "Count": [total_genes, mapped_count, promoter_hit_count]
    }
    cov_df = pd.DataFrame(coverage_data)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Category", y="Count", data=cov_df, palette="viridis")
    plt.title("Validation Coverage", fontsize=14)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_coverage.png", dpi=600)
    plt.close()

    # 2. Histogram of distances from TSS
    plt.figure(figsize=(6, 4))
    sns.histplot(results_df["distance_from_tss"], bins=30, kde=True, color="steelblue")
    plt.title("Distribution of TFBS Distances from TSS", fontsize=14)
    plt.xlabel("Distance from TSS (bp)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_distance_hist.png", dpi=600)
    plt.close()

    # 3. Per-gene hit count
    hit_counts = results_df.groupby("regulated_gene").size().reset_index(name="hit_count")
    plt.figure(figsize=(8, 4))
    sns.barplot(x="regulated_gene", y="hit_count", data=hit_counts, palette="mako")
    plt.xticks(rotation=90)
    plt.title("TFBS Hits per Gene Promoter", fontsize=14)
    plt.xlabel("Gene")
    plt.ylabel("Number of TFBS hits")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_per_gene_hits.png", dpi=600)
    plt.close()

    # 4. Strip plot of promoter hits
    plt.figure(figsize=(8, 4))
    sns.stripplot(data=results_df, x="regulated_gene", y="distance_from_tss", jitter=True, palette="deep")
    plt.axhline(0, color="red", linestyle="--", lw=1)
    plt.title("TFBS Distances from TSS per Gene", fontsize=14)
    plt.xlabel("Gene")
    plt.ylabel("Distance from TSS (bp)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_distance_per_gene.png", dpi=600)
    plt.close()

    print(f"Figures saved with prefix: {out_prefix} (600 dpi PNGs)")


def main(gff_file, gene_file, pred_file, out_prefix, upstream, downstream):
    # Load regulated gene list
    regulated_df = pd.read_excel(gene_file) if gene_file.endswith(".xlsx") else pd.read_csv(gene_file)
    regulated_genes = set(regulated_df['Gene'].str.lower().str.strip())

    # Load GFF
    gff_df = load_gff(gff_file)

    # Load predictions
    pred_df = pd.read_csv(pred_file)
    pred_df.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Check required columns
    required_cols = {"contig", "start", "end"}
    if not required_cols.issubset(set(pred_df.columns)):
        raise ValueError(f"Predictions file must have columns: {required_cols}")

    # Prepare outputs
    promoters = []
    results = []
    unmatched = []
    no_hits = []

    # Iterate regulated genes
    for gene in regulated_genes:
        matches = gff_df[gff_df["gene"] == gene]
        if matches.empty:
            unmatched.append({"regulated_gene": gene})
            continue

        gene_has_hits = False
        for _, gff_row in matches.iterrows():
            if gff_row["strand"] == "+":
                tss = gff_row["start"]
                prom_start = max(1, tss - upstream)
                prom_end = tss + downstream
            else:
                tss = gff_row["end"]
                prom_start = max(1, tss - downstream)
                prom_end = tss + upstream

            promoters.append({
                "regulated_gene": gene,
                "seqid": gff_row["seqid"],
                "strand": gff_row["strand"],
                "tss": tss,
                "promoter_start": prom_start,
                "promoter_end": prom_end
            })

            # Find TFBS hits in promoter
            hits = pred_df[
                (pred_df["contig"] == gff_row["seqid"]) &
                (pred_df["start"] >= prom_start) &
                (pred_df["end"] <= prom_end)
            ]

            for _, hit in hits.iterrows():
                hit_center = (hit["start"] + hit["end"]) // 2
                distance = hit_center - tss if gff_row["strand"] == "+" else tss - hit_center

                results.append({
                    "regulated_gene": gene,
                    "seqid": gff_row["seqid"],
                    "strand": gff_row["strand"],
                    "tss": tss,
                    "promoter_start": prom_start,
                    "promoter_end": prom_end,
                    "hit_start": hit["start"],
                    "hit_end": hit["end"],
                    "hit_center": hit_center,
                    "distance_from_tss": distance,
                    "hit_strand": hit.get("strand", ""),
                    "hit_seq": hit.get("seq", ""),
                    "hit_score": hit.get("score", "")
                })
                gene_has_hits = True

        if not gene_has_hits:
            no_hits.append({"regulated_gene": gene})

    # Save outputs
    results_df = pd.DataFrame(results)
    unmatched_df = pd.DataFrame(unmatched)
    no_hits_df = pd.DataFrame(no_hits)

    results_df.to_csv(f"{out_prefix}_results.csv", index=False)
    unmatched_df.to_csv(f"{out_prefix}_unmatched.csv", index=False)
    no_hits_df.to_csv(f"{out_prefix}_no_hits.csv", index=False)

    # Summary stats
    total_genes = len(regulated_genes)
    mapped_count = total_genes - len(unmatched_df)
    promoter_hit_count = results_df['regulated_gene'].nunique() if not results_df.empty else 0

    print("=== Validation Summary ===")
    print(f"Total regulated genes: {total_genes}")
    print(f"Mapped to GFF: {mapped_count}")
    print(f"With promoter hits: {promoter_hit_count}")
    print(f"Total TFBS hits in promoters: {len(results_df)}")

    perc = 100 * promoter_hit_count / mapped_count if mapped_count > 0 else 0
    print(f"Validation success (based on mapped genes): {perc:.2f}%")
    print(f"\nOutput files written with prefix: {out_prefix}")

    # Generate visualizations
    make_visualizations(results_df, total_genes, mapped_count, promoter_hit_count, out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TFBS predictions")
    parser.add_argument("--gff", required=True, help="Genome annotation GFF file")
    parser.add_argument("--genes", required=True, help="CSV/Excel file with regulated genes (column 'Gene')")
    parser.add_argument("--pred", required=True, help="Predicted TFBS CSV")
    parser.add_argument("--out", required=True, help="Output file prefix")
    parser.add_argument("--upstream", type=int, default=300,
                        help="Promoter upstream window size (default: 300)")
    parser.add_argument("--downstream", type=int, default=50,
                        help="Promoter downstream window size (default: 50)")
    args = parser.parse_args()
    main(args.gff, args.genes, args.pred, args.out, args.upstream, args.downstream)
