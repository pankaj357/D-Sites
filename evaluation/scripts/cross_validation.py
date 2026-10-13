#!/usr/bin/env python3
"""
site-level grouped 5-fold cross-validation of D-Sites (AmrZ / GlxR / CodY; also used for FNR E. coli internal validation).
Usage:
  python cross_validation.py --tf AmrZ [--folds 5 --trees 300 --n_jobs -1]
  python cross_validation.py --smoke          # 2 folds, 50 trees, AmrZ only
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

import eval_common as ec


def run_cv(tf, n_splits=ec.N_SPLITS, n_trees=ec.N_TREES, seed=ec.SEED,
           neg_ratio=ec.NEG_RATIO, feature_idx=None, n_jobs=-1,
           outdir=ec.RESULTS, tag="", save_oof=True, write_json=True,
           verbose=True):
    """Run the grouped-CV protocol. feature_idx optionally restricts the
    feature columns (ablation, Part D). Returns the results dict."""
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = ec.build_bundle(tf)
    if "sites" not in bundle:
        raise SystemExit(f"{tf}: no known sites configured")
    sites = bundle["sites"].reset_index(drop=True)
    X = bundle["X"][:, feature_idx] if feature_idx is not None else bundle["X"]
    win_ov = bundle["win_ov"]
    labels = bundle["labels"]
    site2wins = bundle["site2wins"]
    n_win = len(bundle["win_pos"])
    n_sites = len(sites)

    # ---------------- positives (one best k-mer per site) ----------------
    pwm, k, _ = ec.load_motif_pwm(bundle["cfg"]["motif"])
    X_pos = ec.kmer_features(sites["kmer"].tolist(), pwm, bundle["bg"])
    if feature_idx is not None:
        X_pos = X_pos[:, feature_idx]
    y_pos = np.ones(n_sites, dtype=int)
    groups = sites["site_id"].to_numpy()

    # eligible negative windows: overlap NO known site (padded) at all
    eligible = np.nonzero(labels == 0)[0]

    gkf = GroupKFold(n_splits=n_splits)
    per_fold = []
    fold_of_site = np.full(n_sites, -1, dtype=int)
    # per-fold window probabilities are stored only for the pooled assignment
    fold_probs = np.full((n_splits, n_win), np.nan, dtype=np.float32)

    leakage_ok = True
    for fold, (tr, te) in enumerate(gkf.split(X_pos, y_pos, groups)):
        # ---- hard leakage assertion: no site group in both partitions ----
        inter = set(groups[tr]) & set(groups[te])
        assert not inter, f"LEAKAGE fold {fold}: groups in train AND test: {inter}"
        leakage_ok = leakage_ok and not inter
        fold_of_site[te] = fold

        test_site_ids = groups[te]
        train_site_set = set(groups[tr].tolist())

        # ---- negatives for this fold (seeded) ----
        rng = np.random.default_rng(seed + fold)
        n_neg = neg_ratio * len(tr)
        sel = eligible[rng.choice(len(eligible), size=n_neg, replace=False)]
        neg_rows = np.sort(np.concatenate([2 * sel, 2 * sel + 1]))
        X_tr = np.vstack([X_pos[tr], X[neg_rows]])
        y_tr = np.concatenate([np.ones(len(tr), int), np.zeros(len(neg_rows), int)])

        # ---- train RF (OOB for cutoff selection) ----
        clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs,
                                     random_state=seed, oob_score=True)
        clf.fit(X_tr, y_tr)
        oob = clf.oob_decision_function_[:, 1]
        ok = ~np.isnan(oob)
        cutoff, oob_f1 = ec.f1_cutoff(y_tr[ok], oob[ok])

        # ---- scan ALL promoter windows ----
        prob2 = clf.predict_proba(X)[:, 1].astype(np.float32)
        prob = np.maximum(prob2[0::2], prob2[1::2])      # max over strands
        fold_probs[fold] = prob

        # ---- scored windows: exclude overlaps of TRAIN-fold sites ----
        scored = np.array(
            [not (len(t) and any(s in train_site_set for s in t)) for t in win_ov])
        m = ec.window_site_metrics(prob, labels, scored, test_site_ids,
                                   site2wins, cutoff)
        m.update(fold=fold, n_train_sites=int(len(tr)), n_test_sites_fold=int(len(te)),
                 n_neg_requested=int(n_neg), n_neg_rows=int(len(neg_rows)),
                 oob_f1=float(oob_f1), oob_score=float(clf.oob_score_),
                 leakage_check="PASSED")
        per_fold.append(m)
        if verbose:
            print(f"[{tf}] fold {fold}: cutoff={cutoff:.4f} recall={m['recall']:.3f} "
                  f"precision={m['precision']:.3f} F1={m['F1']:.3f} "
                  f"FPR={m['FPR']:.2e} PR-AUC={m['PR_AUC']:.4f} "
                  f"top5%(P={m['top5pct_precision']:.3f},R={m['top5pct_recall']:.3f})",
                  flush=True)

    # ---------------- pooled out-of-fold predictions ----------------
    rng = np.random.default_rng(seed + 777)
    assign = rng.integers(0, n_splits, size=n_win)
    valid = np.ones(n_win, dtype=bool)
    n_dropped = 0
    for i in np.nonzero(labels == 1)[0]:
        folds = {fold_of_site[s] for s in win_ov[i]}
        if len(folds) == 1:
            assign[i] = folds.pop()
        else:
            valid[i] = False
            n_dropped += 1
    pooled_score = fold_probs[assign, np.arange(n_win)]
    pooled_label = labels
    if save_oof:
        np.savez_compressed(
            outdir / f"oof_{tf}{tag}.npz",
            score=pooled_score[valid], label=pooled_label[valid],
            win_ci=bundle["win_ci"][valid], win_pos=bundle["win_pos"][valid],
            fold=assign[valid], n_dropped_multifold=np.int64(n_dropped))
    pooled_ap = float(average_precision_score(pooled_label[valid],
                                              pooled_score[valid]))

    # ---------------- summary ----------------
    metric_keys = ["recall", "precision", "F1", "FPR", "PR_AUC",
                   "recall_in_space", "top5pct_precision", "top5pct_recall",
                   "top5pct_recall_in_space"]
    summary = {kk: dict(zip(("mean", "sd"), ec.mean_sd(per_fold, kk)))
               for kk in metric_keys}
    summary["cutoff"] = dict(zip(("mean", "sd"), ec.mean_sd(per_fold, "cutoff")))
    results = dict(
        tf=tf, protocol=__doc__, n_sites=int(n_sites), n_windows=int(n_win),
        n_splits=n_splits, n_trees=n_trees, seed=seed, neg_ratio=neg_ratio,
        feature_set=(ec.FEATURE_NAMES if feature_idx is None
                     else [ec.FEATURE_NAMES[i] for i in feature_idx]),
        leakage_assertion="PASSED" if leakage_ok else "FAILED",
        pooled_PR_AUC=pooled_ap,
        pooled_n_windows=int(valid.sum()),
        pooled_dropped_multifold_windows=int(n_dropped),
        per_fold=per_fold, summary=summary,
        runinfo=ec.runinfo(dict(n_jobs=str(n_jobs),
                                wall_seconds=round(time.time() - t0, 1))))
    if write_json:
        ec.save_json(results, outdir / f"cv_{tf}{tag}.json")
    if verbose:
        print(f"[{tf}] DONE in {time.time()-t0:.0f}s | " +
              " ".join(f"{kk}={summary[kk]['mean']:.3f}±{summary[kk]['sd']:.3f}"
                       for kk in ("recall", "precision", "F1", "PR_AUC")),
              flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["AmrZ", "GlxR", "CodY", "FNR_ecoli"])
    ap.add_argument("--folds", type=int, default=ec.N_SPLITS)
    ap.add_argument("--trees", type=int, default=ec.N_TREES)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--outdir", default=str(ec.RESULTS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings: 2 folds, 50 trees, AmrZ only")
    args = ap.parse_args()
    if args.smoke:
        outdir = Path(args.outdir) if args.outdir != str(ec.RESULTS) \
            else ec.RESULTS / "smoke_test"
        run_cv("AmrZ", n_splits=2, n_trees=50, n_jobs=args.n_jobs,
               outdir=outdir, tag="_smoke")
        return
    if not args.tf:
        ap.error("--tf required (or --smoke)")
    run_cv(args.tf, n_splits=args.folds, n_trees=args.trees, n_jobs=args.n_jobs,
           outdir=Path(args.outdir), tag=args.tag)


if __name__ == "__main__":
    main()
