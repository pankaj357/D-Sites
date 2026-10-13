#!/usr/bin/env python3
"""
Precision-recall curves

Outputs:
  Revision/figures/pr_curve_{TF}.png          window-level PR (spec definition)
  Revision/results/pr_curve_{TF}.csv          (grid, full-data precision, band)
  Revision/results/pr_summary.csv             (AP + 95% CI + prevalence, all TFs)
  Revision/figures/pr_sitelevel_{TF}.png      SUPPLEMENTARY site-level view:
                                              x = site recall (fraction of known
                                              sites recovered), y = window-level
                                              precision, swept over the score
                                              threshold. This is the informative
                                              operating characteristic for a
                                              site-discovery tool (window-level
                                              prevalence ~3e-3 makes the
                                              window-level PR curve hug zero by
                                              construction: ~20-26 windows
                                              overlap each padded site but only
                                              1-2 are true motif matches).
                                              Bands: bootstrap over SITES
                                              (1000); precision CI: binomial
                                              normal approximation.
  Revision/results/pr_sitelevel_{TF}.csv

Usage:
  python make_pr_curves.py --all [--boot 1000] [--n_jobs 4]
  python make_pr_curves.py --smoke       # AmrZ only, 20 bootstraps
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, precision_recall_curve

import eval_common as ec

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


def pr_on_grid(scores, labels):
    prec, rec, _ = precision_recall_curve(labels, scores)
    return np.interp(GRID, rec[::-1], prec[::-1])


def _boot_rep(b, scores, labels, seed):
    rng = np.random.default_rng(seed * 100003 + b)
    n = len(labels)
    idx = rng.integers(0, n, n)
    y, s = labels[idx], scores[idx]
    if y.sum() == 0:
        return np.full(len(GRID), np.nan), np.nan
    prec, rec, _ = precision_recall_curve(y, s)
    return np.interp(GRID, rec[::-1], prec[::-1]), \
        float(average_precision_score(y, s))


def curve_with_band(scores, labels, n_boot, seed, n_jobs):
    full = pr_on_grid(scores, labels)
    ap = float(average_precision_score(labels, scores))
    reps = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_boot_rep)(b, scores, labels, seed) for b in range(n_boot))
    band = np.array([r[0] for r in reps])
    aps = np.array([r[1] for r in reps])
    return dict(full=full, ap=ap,
                lo=np.nanpercentile(band, 2.5, axis=0),
                hi=np.nanpercentile(band, 97.5, axis=0),
                ap_lo=float(np.nanpercentile(aps, 2.5)),
                ap_hi=float(np.nanpercentile(aps, 97.5)),
                prevalence=float(labels.mean()))


def sitelevel_curves(tf, n_boot, seed, outdir, figdir, verbose=True):
    """Supplementary site-recall vs window-precision operating curve.
    For each method: site score = max window score over the site's padded
    region (D-Sites: pooled OOF; FIMO: global run); thresholds swept over a
    shared quantile grid. recall(t) = fraction of sites with site score >= t
    (reported over ALL sites and over the in-space subset); precision(t) =
    fraction of windows with score >= t that overlap a known site."""
    bundle = ec.build_bundle(tf)
    sites = bundle["sites"]
    labels = bundle["labels"]
    site2wins = bundle["site2wins"]
    n_win = len(labels)
    all_sids = sites["site_id"].to_numpy()
    in_space = np.array([sid in site2wins for sid in all_sids])

    oof = np.load(Path(outdir) / f"oof_{tf}.npz")
    fim = np.load(Path(outdir) / f"fimo_scores_{tf}.npz")
    # align D-Sites pooled OOF scores to the universe (unscored -> -inf)
    key_all = (bundle["win_ci"].astype(np.int64) << 40) | bundle["win_pos"].astype(np.int64)
    key_oof = (oof["win_ci"].astype(np.int64) << 40) | oof["win_pos"].astype(np.int64)
    order = np.argsort(key_all, kind="stable")
    pos = np.searchsorted(key_all[order], key_oof)
    ds_score = np.full(n_win, -np.inf)
    ds_score[order[pos]] = oof["score"]
    f_score = fim["score"].astype(np.float64)
    f_score[f_score == 0] = -np.inf          # unreported windows never pass

    def site_max(score):
        sm = np.full(len(all_sids), -np.inf)
        for i, sid in enumerate(all_sids):
            w = site2wins.get(int(sid))
            if w is not None:
                sm[i] = np.max(score[w])
        return sm

    curves = {}
    for method, sc in (("D-Sites", ds_score), ("FIMO", f_score)):
        sm = site_max(sc)
        finite = sc[np.isfinite(sc)]
        thr = np.quantile(finite, np.linspace(0.9999, 0.0, 400))
        thr = np.unique(thr)[::-1]
        lb = labels.astype(bool)
        tp = np.empty(len(thr), dtype=np.int64)
        pred = np.empty(len(thr), dtype=np.int64)
        for lo in range(0, len(thr), 50):# chunk to bound memory
            ge = sc[None, :] >= thr[lo:lo + 50, None]
            tp[lo:lo + 50] = (ge & lb[None, :]).sum(axis=1)
            pred[lo:lo + 50] = ge.sum(axis=1)
        prec = np.divide(tp, pred, out=np.zeros(len(thr), dtype=float),
                         where=pred > 0)
        # binomial normal-approx 95% CI for precision
        pz = 1.96 * np.sqrt(np.clip(prec * (1 - prec) / np.clip(pred, 1, None),
                                    0, None))
        rec_all = (sm[None, :] >= thr[:, None]).mean(axis=1)
        rec_is = (sm[None, in_space] >= thr[:, None]).mean(axis=1)
        # bootstrap over sites for recall band (in-space recall)
        rng = np.random.default_rng(seed)
        n_is = int(in_space.sum())
        boot = np.empty((n_boot, len(thr)))
        sm_is = sm[in_space]
        for b in range(n_boot):
            idx = rng.integers(0, n_is, n_is)
            boot[b] = (sm_is[idx][None, :] >= thr[:, None]).mean(axis=1)
        curves[method] = dict(thr=thr, precision=prec,
                              prec_lo=np.clip(prec - pz, 0, 1),
                              prec_hi=np.clip(prec + pz, 0, 1),
                              recall_all=rec_all, recall_in_space=rec_is,
                              rec_lo=np.percentile(boot, 2.5, axis=0),
                              rec_hi=np.percentile(boot, 97.5, axis=0))
    # CSV + figure
    rows = []
    for method, c in curves.items():
        for i, t in enumerate(c["thr"]):
            rows.append(dict(TF=tf, method=method, threshold=t,
                             precision=c["precision"][i],
                             precision_lo95=c["prec_lo"][i],
                             precision_hi95=c["prec_hi"][i],
                             recall_all_sites=c["recall_all"][i],
                             recall_in_space=c["recall_in_space"][i],
                             recall_lo95=c["rec_lo"][i],
                             recall_hi95=c["rec_hi"][i]))
    pd.DataFrame(rows).to_csv(Path(outdir) / f"pr_sitelevel_{tf}.csv",
                              index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"D-Sites": "#1f77b4", "FIMO": "#d62728"}
    for method, c in curves.items():
        ax.plot(c["recall_in_space"], c["precision"], color=colors[method],
                lw=2, label=method)
        ax.fill_between(c["recall_in_space"], c["prec_lo"], c["prec_hi"],
                        color=colors[method], alpha=0.15)
    ax.set_xlabel("site recall (known sites recovered, in-search-space subset)")
    ax.set_ylabel("window precision (predictions overlapping a known site)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{tf}: held-out site-level operating curve\n"
                 f"(D-Sites pooled 5-fold CV vs FIMO, identical search space)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(figdir) / f"pr_sitelevel_{tf}.png", dpi=300)
    plt.close(fig)
    if verbose:
        print(f"[{tf}] site-level curve saved", flush=True)


def make_tf(tf, n_boot=1000, seed=ec.SEED, n_jobs=4, outdir=ec.RESULTS,
            figdir=ec.FIGURES, verbose=True):
    t0 = time.time()
    outdir, figdir = Path(outdir), Path(figdir)
    oof = np.load(outdir / f"oof_{tf}.npz")
    fim = np.load(outdir / f"fimo_scores_{tf}.npz")
    res = {}
    for method, sc, lb in (("D-Sites", oof["score"], oof["label"]),
                           ("FIMO", fim["score"], fim["label"])):
        res[method] = curve_with_band(sc.astype(np.float64), lb.astype(np.int8),
                                      n_boot, seed, n_jobs)
        if verbose:
            r = res[method]
            print(f"[{tf}] {method}: AP={r['ap']:.4f} "
                  f"(95% CI {r['ap_lo']:.4f}-{r['ap_hi']:.4f}) "
                  f"prevalence={r['prevalence']:.2e}", flush=True)
    # CSV of underlying data
    rows = []
    for method in ("D-Sites", "FIMO"):
        r = res[method]
        for i, g in enumerate(GRID):
            rows.append(dict(TF=tf, method=method, recall=g,
                             precision_full=r["full"][i],
                             precision_lo95=r["lo"][i],
                             precision_hi95=r["hi"][i],
                             AP=r["ap"], AP_lo95=r["ap_lo"], AP_hi95=r["ap_hi"],
                             prevalence=r["prevalence"]))
    pd.DataFrame(rows).to_csv(outdir / f"pr_curve_{tf}.csv", index=False)
    sitelevel_curves(tf, n_boot, seed, outdir, figdir, verbose=verbose)
    # figure
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"D-Sites": "#1f77b4", "FIMO": "#d62728"}
    for method in ("D-Sites", "FIMO"):
        r = res[method]
        ax.plot(GRID, r["full"], color=colors[method], lw=2,
                label=f"{method} (AP={r['ap']:.3f}, 95% CI "
                      f"{r['ap_lo']:.3f}\u2013{r['ap_hi']:.3f})")
        ax.fill_between(GRID, r["lo"], r["hi"], color=colors[method], alpha=0.2)
    prev = res["D-Sites"]["prevalence"]
    ax.axhline(prev, color="gray", ls="--", lw=1,
               label=f"no-skill (prevalence={prev:.2e})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{tf}: held-out PR curve, identical promoter search space\n"
                 f"(D-Sites pooled 5-fold CV vs FIMO global run)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / f"pr_curve_{tf}.png", dpi=300)
    plt.close(fig)
    if verbose:
        print(f"[{tf}] figure + CSV saved ({time.time()-t0:.0f}s)", flush=True)
    return {m: dict(AP=res[m]["ap"], AP_lo95=res[m]["ap_lo"],
                    AP_hi95=res[m]["ap_hi"], prevalence=res[m]["prevalence"])
            for m in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", action="append",
                    choices=["AmrZ", "GlxR", "CodY", "FNR_ecoli"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        make_tf("AmrZ", n_boot=20, n_jobs=args.n_jobs,
                outdir=ec.RESULTS / "smoke_test",
                figdir=ec.RESULTS / "smoke_test")
        return
    tfs = (["AmrZ", "GlxR", "CodY"] if args.all else (args.tf or []))
    if not tfs:
        ap.error("--tf or --all required")
    summary = []
    for tf in tfs:
        r = make_tf(tf, n_boot=args.boot, n_jobs=args.n_jobs)
        for method, d in r.items():
            summary.append(dict(TF=tf, method=method, **d))
    pd.DataFrame(summary).to_csv(ec.RESULTS / "pr_summary.csv", index=False)
    print(f"pr_summary.csv written for {tfs}", flush=True)


if __name__ == "__main__":
    main()
