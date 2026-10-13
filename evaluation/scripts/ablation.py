#!/usr/bin/env python3
"""
Same grouped 5-fold CV protocol as cross_validation.run_cv applied to three
feature sets:
  1. PWM_only          — PWM log-odds only                    (1 feature)
  2. PWM_GC_Entropy    — PWM log-odds + GC content + entropy  (3 features)
  3. all_11            — the full hybrid vector               (11 features:
                         + mean/std of MGW, PropT, Roll, HelT)

Outputs:
  results/ablation_{TF}.json   per-feature-set per-fold and mean±SD metrics
                               (focus: PR-AUC and recall@top5%)
  Revision/figures/ablation.png  grouped bar chart (panels: PR-AUC,
                               recall@top5%; groups: TF; bars: feature set;
                               error bars: SD across the 5 folds)

Usage:
  python ablation.py --all [--folds 5 --trees 300 --n_jobs -1]
  python ablation.py --smoke     # 2 folds, 50 trees, AmrZ only
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_common as ec
from cross_validation import run_cv

# metrics shown in the figure (spec: PR-AUC and recall@top5%)
FIG_METRICS = [("PR_AUC", "PR-AUC (average precision)"),
               ("top5pct_recall", "recall @ top 5%")]
JSON_METRICS = ["PR_AUC", "recall", "precision", "F1", "FPR",
                "recall_in_space", "top5pct_precision", "top5pct_recall",
                "top5pct_recall_in_space"]


def run_ablation_tf(tf, n_splits, n_trees, n_jobs, outdir, verbose=True):
    t0 = time.time()
    out = dict(tf=tf, protocol="identical grouped CV folds/seeds as Part A; "
                               "only the feature set varies",
               feature_sets={}, runinfo=ec.runinfo())
    for name, idx in ec.ABLATION_SETS.items():
        r = run_cv(tf, n_splits=n_splits, n_trees=n_trees,
                   feature_idx=idx, n_jobs=n_jobs, outdir=outdir,
                   save_oof=False, write_json=False, verbose=verbose)
        out["feature_sets"][name] = dict(
            features=r["feature_set"], summary=r["summary"],
            per_fold=r["per_fold"])
        if verbose:
            s = r["summary"]
            print(f"[{tf}|{name}] PR-AUC={s['PR_AUC']['mean']:.4f}±"
                  f"{s['PR_AUC']['sd']:.4f} "
                  f"recall@top5%={s['top5pct_recall']['mean']:.3f}±"
                  f"{s['top5pct_recall']['sd']:.3f}", flush=True)
    out["wall_seconds"] = round(time.time() - t0, 1)
    ec.save_json(out, Path(outdir) / f"ablation_{tf}.json")
    return out


def make_figure(outdir=ec.RESULTS, figdir=ec.FIGURES, tfs=("AmrZ", "GlxR", "CodY")):
    data = {}
    for tf in tfs:
        p = Path(outdir) / f"ablation_{tf}.json"
        if p.exists():
            import json
            data[tf] = json.load(open(p))
    if not data:
        print("no ablation jsons found; figure skipped", flush=True)
        return
    sets = list(ec.ABLATION_SETS.keys())
    colors = {"PWM_only": "#7fc97f", "PWM_GC_Entropy": "#fdc086",
              "all_11": "#386cb0"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.25
    for ax, (mkey, mlabel) in zip(axes, FIG_METRICS):
        x = np.arange(len(data))
        for si, sname in enumerate(sets):
            means, sds = [], []
            for tf in data:
                summ = data[tf]["feature_sets"][sname]["summary"][mkey]
                means.append(summ["mean"])
                sds.append(summ["sd"])
            ax.bar(x + (si - 1) * width, means, width, yerr=sds, capsize=4,
                   label=sname, color=colors[sname], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(list(data.keys()))
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel} by feature set (mean ± SD, 5-fold grouped CV)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("D-Sites feature ablation — held-out evaluation "
                 "(identical folds, seeds and search space)", y=1.02)
    fig.tight_layout()
    Path(figdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figdir) / "ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("ablation.png written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", action="append", choices=["AmrZ", "GlxR", "CodY"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--folds", type=int, default=ec.N_SPLITS)
    ap.add_argument("--trees", type=int, default=ec.N_TREES)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_ablation_tf("AmrZ", n_splits=2, n_trees=50, n_jobs=args.n_jobs,
                        outdir=ec.RESULTS / "smoke_test")
        return
    tfs = (["AmrZ", "GlxR", "CodY"] if args.all else (args.tf or []))
    if not tfs:
        ap.error("--tf or --all required")
    for tf in tfs:
        run_ablation_tf(tf, args.folds, args.trees, args.n_jobs, ec.RESULTS)
    make_figure()


if __name__ == "__main__":
    main()
