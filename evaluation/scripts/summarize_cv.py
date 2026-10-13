#!/usr/bin/env python3
"""
Build the combined results/cv_summary.csv from the per-TF results/cv_{TF}.json files (and cv_FNR_ecoli.json if present).

Wide format: one row per TF; mean ± SD across the 5 grouped CV folds for every
protocol metric, plus search-space coverage counts and pooled-OOF PR-AUC.
"""

import json
from pathlib import Path

import pandas as pd

import eval_common as ec

METRICS = ["recall", "recall_in_space", "precision", "F1", "FPR", "PR_AUC",
           "top5pct_precision", "top5pct_recall", "top5pct_recall_in_space",
           "cutoff"]


def main():
    rows = []
    for tf in ("AmrZ", "GlxR", "CodY", "FNR_ecoli"):
        p = ec.RESULTS / f"cv_{tf}.json"
        if not p.exists():
            print(f"skip {tf}: {p} missing")
            continue
        j = json.load(open(p))
        row = dict(TF=tf, n_sites=j["n_sites"], n_windows=j["n_windows"],
                   n_sites_in_space=sum(f["n_test_sites_in_space"]
                                        for f in j["per_fold"]),
                   pooled_PR_AUC=j["pooled_PR_AUC"],
                   leakage_assertion=j["leakage_assertion"],
                   n_splits=j["n_splits"], n_trees=j["n_trees"], seed=j["seed"])
        for m in METRICS:
            row[f"{m}_mean"] = j["summary"][m]["mean"]
            row[f"{m}_sd"] = j["summary"][m]["sd"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = ec.RESULTS / "cv_summary.csv"
    df.to_csv(out, index=False)
    print(df[["TF", "n_sites", "n_sites_in_space", "recall_mean",
              "recall_in_space_mean", "precision_mean", "F1_mean",
              "PR_AUC_mean", "top5pct_recall_mean"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
