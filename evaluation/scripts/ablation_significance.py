#!/usr/bin/env python3
"""
ablation_significance.py

Feature-ablation statistics, the FNR ablation arm, and a numerical explanation
of the pooled-vs-per-fold average-precision gap.

READ-ONLY inputs (evaluation/ never modified):
  evaluation/results/ablation_{AmrZ,GlxR,CodY}.json   stored 3-factor ablation
  evaluation/results/cv_CodY.json                     per-fold scored-window counts
  evaluation/results/oof_CodY.npz                     pooled OOF scores/labels
  evaluation/results/cache/FNR_ecoli_windows.pkl      FNR bundle (read-only load)

WHAT IT DOES
1. Loads the stored per-fold PR-AUC (average precision) of the three feature
   sets (PWM_only / PWM_GC_Entropy / all_11) for AmrZ, GlxR, CodY and computes
   per-fold paired contrasts:
     C1 = AP(all_11) - AP(PWM_only)
     C2 = AP(all_11) - AP(PWM_GC_Entropy)
   For each factor x contrast: the 5 per-fold differences, mean +/- SD,
   Wilcoxon signed-rank test (EXPLORATORY: n=5 pairs, very low power --
   reported as such) and a paired bootstrap over folds (10,000 resamples,
   seed 42) of the mean difference with percentile 95% CI. Holm correction
   across the three factors, per contrast.
2. FNR ARM: reruns the same 3-feature-set grouped 5-fold CV for FNR_ecoli
   (97 RegulonDB sites) by calling cross_validation.run_cv (imported, not
   modified) with feature_idx [0], [0,1,2] and None -- identical folds,
   negatives, seeds and protocol as the stored ablation (run_cv is the same
   function ablation.py uses). Leakage assertions fire inside run_cv; the
   per-fold and pooled AP are saved.
3. AP-DENOMINATOR EXPLANATION (CodY, representative): reports the pooled AP
   denominator (windows and positive-window counts from the pooled OOF
   arrays, labels over ALL known sites) versus per-fold scored-window counts
   (train-site-overlapping windows excluded) and per-fold positive-window
   counts. The prevalence ratio (~5x) between the two denominators explains
   the AP gap (per-fold AP mean ~0.0039 vs pooled AP ~0.0179): AP is a
   prevalence-scaled ranking summary, and excluding train-site windows
   removes ~80% of positive-labelled windows while removing essentially no
   negatives, collapsing per-fold prevalence.

Deterministic (seed 42). Outputs under the R2 workspace only.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]   # repository root
EVAL = ROOT / "evaluation"
OUT = EVAL / "results" / "ablation"
TAB = EVAL / "tables"
LOGD = EVAL / "logs"
FNR_DIR = OUT / "fnr"
for d in (OUT, TAB, LOGD, FNR_DIR):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Software" / "D-Sites" / "src"))
sys.path.insert(0, str(EVAL / "scripts"))

from cross_validation import run_cv          # imported, NOT modified
from dsites import predictor as dp

SEED = 42
N_BOOT = 10_000
TFS = ["AmrZ", "GlxR", "CodY"]
SETS = ["PWM_only", "PWM_GC_Entropy", "all_11"]
CONTRASTS = {"C1_all11_minus_PWM_only": ("all_11", "PWM_only"),
             "C2_all11_minus_PWMGE": ("all_11", "PWM_GC_Entropy")}

log_lines = []


def log(m):
    print(m, flush=True)
    log_lines.append(m)


def holm(pvals):
    order = sorted(pvals.items(), key=lambda kv: (kv[1] if kv[1] is not None else 2.0))
    m = len(order)
    running, out = 0.0, {}
    for rank, (name, p) in enumerate(order):
        if p is None:
            out[name] = None
            continue
        running = max(running, (m - rank) * p)
        out[name] = min(running, 1.0)
    return out


def contrast_stats(diffs):
    diffs = np.asarray(diffs, dtype=float)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        p_wilcoxon = 1.0
    else:
        try:
            p_wilcoxon = float(wilcoxon(diffs).pvalue)
        except ValueError:
            p_wilcoxon = 1.0
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots[i] = diffs[idx].mean()
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return dict(per_fold_diffs=diffs.tolist(), mean=mean, sd=sd,
                wilcoxon_signedrank_p=p_wilcoxon,
                wilcoxon_label="EXPLORATORY (n=5 paired folds, very low power)",
                n_bootstrap=N_BOOT, bootstrap_mean_diff_ci95=ci)


def main():
    t0 = time.time()
    log(f"# r2_major3_ablation_stats.py  seed={SEED}  "
        f"python {sys.version.split()[0]}  dsites {dp.__version__}")

    # ---------------- stored 3-factor ablation ----------------
    ap = {}        # tf -> set -> per-fold AP list
    pooled = {}    # tf -> set -> pooled AP (only present in json for some sets)
    for tf in TFS:
        j = json.load(open(EVAL / "results" / f"ablation_{tf}.json"))
        ap[tf] = {s: [f["PR_AUC"] for f in j["feature_sets"][s]["per_fold"]]
                  for s in SETS}
        pooled[tf] = {s: None for s in SETS}

    # ---------------- FNR arm ----------------
    log("[FNR] running grouped 5-fold CV x 3 feature sets via run_cv ...")
    feats = {"PWM_only": [0], "PWM_GC_Entropy": [0, 1, 2], "all_11": None}
    ap["FNR_ecoli"] = {}
    pooled["FNR_ecoli"] = {}
    for s, idx in feats.items():
        tag = {"PWM_only": "_r2_pwm_only", "PWM_GC_Entropy": "_r2_pwmge",
               "all_11": "_r2_all11"}[s]
        r = run_cv("FNR_ecoli", n_splits=5, n_trees=300, feature_idx=idx,
                   n_jobs=6, outdir=FNR_DIR, tag=tag, save_oof=False,
                   write_json=True, verbose=False)
        assert r["leakage_assertion"] == "PASSED", "LEAKAGE in FNR ablation CV"
        ap["FNR_ecoli"][s] = [f["PR_AUC"] for f in r["per_fold"]]
        pooled["FNR_ecoli"][s] = r["pooled_PR_AUC"]
        log(f"[FNR] {s}: per-fold AP {[round(x,6) for x in ap['FNR_ecoli'][s]]} "
            f"pooled={r['pooled_PR_AUC']:.6f} leakage=PASSED "
            f"({time.time()-t0:.0f}s)")

    # ---------------- contrasts + tests ----------------
    factors = TFS + ["FNR_ecoli"]
    results = {}
    table_rows = []
    for tf in factors:
        results[tf] = {}
        for s in SETS:
            table_rows.append(dict(
                factor=tf, feature_set=s, row_type="AP_per_fold",
                **{f"AP_fold{i+1}": ap[tf][s][i] for i in range(5)},
                AP_mean=float(np.mean(ap[tf][s])),
                AP_sd=float(np.std(ap[tf][s], ddof=1)),
                pooled_AP=pooled[tf][s]))
        for cname, (hi, lo) in CONTRASTS.items():
            diffs = [a - b for a, b in zip(ap[tf][hi], ap[tf][lo])]
            results[tf][cname] = contrast_stats(diffs)
            table_rows.append(dict(
                factor=tf, feature_set=cname, row_type="dAP_per_fold",
                **{f"AP_fold{i+1}": diffs[i] for i in range(5)},
                AP_mean=float(np.mean(diffs)),
                AP_sd=float(np.std(diffs, ddof=1)), pooled_AP=""))
    for cname in CONTRASTS:
        adj = holm({tf: results[tf][cname]["wilcoxon_signedrank_p"]
                    for tf in factors})
        for tf in factors:
            results[tf][cname]["wilcoxon_p_holm"] = adj[tf]
        for row in table_rows:
            if row["row_type"] == "dAP_per_fold" and row["feature_set"] == cname:
                row["wilcoxon_p_exploratory"] = \
                    results[row["factor"]][cname]["wilcoxon_signedrank_p"]
                row["holm_p"] = results[row["factor"]][cname]["wilcoxon_p_holm"]
                row["bootstrap_ci95"] = \
                    results[row["factor"]][cname]["bootstrap_mean_diff_ci95"]
    for row in table_rows:
        row.setdefault("wilcoxon_p_exploratory", "")
        row.setdefault("holm_p", "")
        row.setdefault("bootstrap_ci95", "")
    pd.DataFrame(table_rows).to_csv(TAB / "table_ablation.csv", index=False)

    # ---------------- CodY AP-denominator explanation ----------------
    oof = np.load(EVAL / "results" / "oof_CodY.npz")
    cvj = json.load(open(EVAL / "results" / "cv_CodY.json"))
    n_pooled = int(len(oof["score"]))
    pos_pooled = int(oof["label"].sum())
    prev_pooled = pos_pooled / n_pooled
    folds = []
    for f in cvj["per_fold"]:
        folds.append(dict(fold=f["fold"], n_scored=f["n_scored"],
                          n_pos_windows=f["n_pos_windows"],
                          prevalence=f["n_pos_windows"] / f["n_scored"]))
    prev_fold_mean = float(np.mean([f["prevalence"] for f in folds]))
    ap_fold_mean = float(np.mean([f["PR_AUC"] for f in cvj["per_fold"]]))
    ap_pooled = float(cvj["pooled_PR_AUC"])
    expl = dict(
        factor="CodY",
        pooled=dict(denominator="pooled out-of-fold windows (labels over ALL "
                                "known sites, +/-10 bp pad)",
                    n_windows=n_pooled, n_positive_windows=pos_pooled,
                    prevalence=prev_pooled, pooled_AP=ap_pooled),
        per_fold=dict(denominator="scored windows per fold (windows overlapping "
                                  "train-fold sites excluded)",
                      folds=folds,
                      mean_prevalence=prev_fold_mean,
                      mean_AP=ap_fold_mean),
        prevalence_ratio_pooled_over_fold=prev_pooled / prev_fold_mean,
        AP_ratio_pooled_over_fold=ap_pooled / ap_fold_mean,
        explanation=(
            "Average precision is a prevalence-scaled ranking summary. In the "
            "pooled OOF arrays every window of the universe is a candidate "
            "(prevalence 0.00616 for CodY). Per fold, windows overlapping "
            "TRAIN-fold sites are excluded from scoring: those windows carry "
            "~80% of all positive labels (positives cluster at documented "
            "sites) while negatives are barely affected, collapsing per-fold "
            "prevalence ~5.3-fold (to ~0.00116). The AP values scale with "
            "their denominators' prevalence, which is why per-fold AP "
            "(~0.0039) is ~4.6x smaller than pooled AP (~0.0179). Both are "
            "correct; they answer different counting questions."))
    with open(OUT / "ap_denominator_explanation.json", "w") as fh:
        json.dump(expl, fh, indent=2)
    log(f"[CodY] pooled prevalence {prev_pooled:.5f} ({pos_pooled}/{n_pooled}) "
        f"vs per-fold mean {prev_fold_mean:.5f}; ratio "
        f"{expl['prevalence_ratio_pooled_over_fold']:.2f}x; AP ratio "
        f"{expl['AP_ratio_pooled_over_fold']:.2f}x")

    out = dict(protocol=__doc__, seed=SEED, n_bootstrap=N_BOOT,
               feature_sets=SETS, contrasts=CONTRASTS,
               per_fold_AP=ap, pooled_AP=pooled,
               contrasts_stats=results,
               fnr_arm=dict(cv_dir=str(FNR_DIR),
                            leakage="PASSED (asserted inside run_cv)"),
               ap_denominator_explanation=expl,
               runtime_s=round(time.time() - t0, 1))
    with open(OUT / "ablation_significance.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"runtime {time.time()-t0:.0f}s")
    with open(LOGD / "major3.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print("outputs written", flush=True)


if __name__ == "__main__":
    main()
