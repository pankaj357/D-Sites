#!/usr/bin/env python3
"""
r2_major1_matched_pairs.py — EVB-25-0116 R2, Major Comment 1.

Reproduces, INDEPENDENTLY from the cached evaluation artifacts, the
matched-count site-recovery comparison between D-Sites (pooled out-of-fold
probabilities, grouped 5-fold CV) and FIMO (one global run on the identical
promoter search space), and adds PAIRED site-level statistics.

READ-ONLY inputs (nothing in evaluation/ is modified):
  evaluation/results/cache/{TF}_windows.pkl   window universe, sites,
                                              site->window overlaps (pad 10)
  evaluation/results/oof_{TF}.npz             pooled out-of-fold D-Sites
                                              window scores (cross_validation.py)
  evaluation/results/fimo_scores_{TF}.npz     per-window -log10(p) FIMO scores
                                              (run_fimo_fair.py)
  evaluation/results/fimo_fair_{TF}.json      stored comparison (verification)

Protocol (identical to evaluation/scripts/run_fimo_fair.py and
cross_validation.py; definitions quoted from those script headers):
  - N = number of FIMO prediction windows at FIMO's default threshold
    (p < 1e-4 <=> score = -log10(p) >= 4).
  - D-Sites set: top-N windows by pooled out-of-fold probability
    (each window contributes exactly one score, from the fold in which its
    overlapped site(s) were held out; windows overlapping sites from multiple
    folds are dropped from pooling -- count reported).
  - FIMO set: top-N windows by -log10(p) (== the p<1e-4 set).
  - A known site is RECOVERED by a method if any selected window overlaps the
    site's coords +/-10 bp pad on the same contig (window = motif-length
    genomic interval; per-window best p over strands; window probability =
    max over strand orientations).
  - IN-SPACE sites (single harmonized definition): site interval +/-10 bp
    intersects any promoter window of the scan universe (i.e. the site has a
    site2wins entry). Recall and all paired statistics use in-space sites.
  - Site-level precision at N = recovered_sites / N.

Paired statistics (sites = independent units; window-overlap dependence is
handled by pairing per site and by the cluster bootstrap):
  - b = sites recovered by D-Sites only; c = FIMO only; exact McNemar
    two-sided p = scipy.stats.binomtest(b, b+c, 0.5) (skipped if b+c == 0).
  - Conditional odds ratio OR = b/c with exact CI: invert two one-sided
    binomial tests at alpha 0.025 (Clopper-Pearson bounds for p = b/(b+c),
    mapped to the odds scale p/(1-p)); open bounds when b==0 or c==0.
  - Paired cluster bootstrap, 10,000 resamples (seed 42): resample in-space
    sites with replacement; per resample compute delta_sites (D-recovered -
    F-recovered) and delta_recall (delta_sites / K); percentile 95% CI.
  - Wilson score 95% CI for each method's in-space site recall and for
    site-level precision at N.
  - Holm correction of the McNemar p-values across the four factors.
  - Reference: reverse top-5%-count comparison (N2 = 5% of pooled windows;
    both methods' top-N2), recomputed from the same npz caches.

No model retraining; deterministic (seed 42). Outputs under the R2 workspace
only (results/r2/, tables/r2/, logs/r2/).
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest

# --------------------------------------------------------------------------
# paths (read-only evaluation/ + R2 workspace outputs)
# --------------------------------------------------------------------------
R2 = Path(__file__).resolve().parents[1]
ROOT = R2.parents[1]
EVAL_RESULTS = ROOT / "evaluation" / "results"
EVAL_SCRIPTS = ROOT / "evaluation" / "scripts"
OUT_RES = R2 / "results" / "r2"
OUT_TAB = R2 / "tables" / "r2"
OUT_LOG = R2 / "logs" / "r2"
for d in (OUT_RES, OUT_TAB, OUT_LOG):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Software" / "D-Sites" / "src"))   # dsites
sys.path.insert(0, str(EVAL_SCRIPTS))                            # eval_common

import eval_common as ec                  # read-only (constants/provenance)
from dsites import predictor as dp

TFS = ["AmrZ", "GlxR", "CodY", "FNR_ecoli"]
SEED = 42
PAD = ec.PAD
N_BOOT = 10_000
FIMO_CUT = -np.log10(1e-4)              # 4.0

# reviewer's table (Major Comment 1)
REVIEWER = {"AmrZ": (7, 18, 483), "GlxR": (13, 17, 185),
            "CodY": (64, 57, 500), "FNR_ecoli": (13, 36, 532)}

KEYSHIFT = 40


def load_bundle(tf):
    """Read-only cache load (mirrors eval_common.cache_path)."""
    with open(EVAL_RESULTS / "cache" / f"{tf}_windows.pkl", "rb") as fh:
        return pickle.load(fh)


def universe_index(bundle):
    return (bundle["win_ci"].astype(np.int64) << KEYSHIFT) | \
        bundle["win_pos"].astype(np.int64)


def align_oof(bundle, tf):
    """Map pooled-OOF rows to universe window indices (searchsorted, as in
    run_fimo_fair.py)."""
    oof = np.load(EVAL_RESULTS / f"oof_{tf}.npz")
    key_all = universe_index(bundle)
    order = np.argsort(key_all, kind="stable")
    key_oof = (oof["win_ci"].astype(np.int64) << KEYSHIFT) | \
        oof["win_pos"].astype(np.int64)
    pos = np.searchsorted(key_all[order], key_oof)
    widx = order[pos]
    return oof["score"].astype(np.float64), oof["label"].astype(np.int8), \
        oof["win_ci"].astype(np.int64), oof["win_pos"].astype(np.int64), widx


def recovered_per_site(sel_mask, bundle, in_space_ids):
    """Boolean per site: any selected window overlaps the padded site."""
    s2w = bundle["site2wins"]
    return np.array([bool(sel_mask[s2w[int(s)]].any()) for s in in_space_ids])


def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(p), float(c - half), float(c + half))


def or_exact_ci(b, c):
    """Conditional McNemar OR = b/c with CI from inverting two one-sided
    binomial tests at 0.025: exact (Clopper-Pearson) bounds for p=b/(b+c),
    mapped to odds p/(1-p). Open bounds when b==0 or c==0."""
    from scipy.optimize import brentq
    k = b + c
    if k == 0:
        return None
    if b == 0:
        p_lo = 0.0
    else:
        p_lo = brentq(lambda p: binom.sf(b - 1, k, p) - 0.025, 1e-12, 1 - 1e-12)
    if c == 0:
        p_hi = 1.0
    else:
        p_hi = brentq(lambda p: binom.cdf(b, k, p) - 0.025, 1e-12, 1 - 1e-12)
    lo = 0.0 if p_lo == 0 else p_lo / (1 - p_lo)
    hi = np.inf if p_hi == 1 else p_hi / (1 - p_hi)
    return (float(lo), float(hi))


def paired_stats(ds_rec, fi_rec, n_boot=N_BOOT, seed=SEED):
    b = int(np.sum(ds_rec & ~fi_rec))
    c = int(np.sum(~ds_rec & fi_rec))
    K = len(ds_rec)
    if b + c > 0:
        p_exact = float(binomtest(b, b + c, 0.5).pvalue)
    else:
        p_exact = None
    orci = or_exact_ci(b, c)
    or_val = (b / c) if c > 0 else (np.inf if b > 0 else np.nan)
    rng = np.random.default_rng(seed)
    d_sites = np.empty(n_boot)
    d_recall = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, K, K)
        d = int(ds_rec[idx].sum()) - int(fi_rec[idx].sum())
        d_sites[i] = d
        d_recall[i] = d / K
    def ci(x):
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return dict(K_in_space=K, b_ds_only=b, c_fimo_only=c,
                mcnemar_exact_two_sided_p=p_exact,
                OR_b_over_c=(None if np.isnan(or_val) else float(or_val)),
                OR_ci95=orci, n_bootstrap=n_boot,
                bootstrap_delta_sites_ci95=ci(d_sites),
                bootstrap_delta_recall_ci95=ci(d_recall))


def method_stats(sel_mask, bundle, in_space_ids, N):
    rec = recovered_per_site(sel_mask, bundle, in_space_ids)
    r, lo, hi = wilson(int(rec.sum()), len(rec))
    prec, plo, phi = wilson(int(rec.sum()), N)
    return dict(recovered_sites=int(rec.sum()), in_space_sites=len(rec),
                recall=r, recall_wilson95=[lo, hi],
                site_precision_at_N=prec, precision_wilson95=[plo, phi]), rec


def topn_mask(full_scores, N):
    """Top-N selection over a full-universe score array (-inf = not scored;
    unscored windows are never selected because N <= #scored)."""
    top = np.argsort(-full_scores, kind="stable")[:N]
    m = np.zeros(len(full_scores), dtype=bool)
    m[top] = True
    return m


def main():
    t0 = time.time()
    repro, paired, reverse_rows = {}, {}, {}
    log = [f"# r2_major1_matched_pairs.py — EVB-25-0116 R2 Major Comment 1",
           f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
           f"scipy {__import__('scipy').__version__}  sklearn "
           f"{__import__('sklearn').__version__}  dsites {dp.__version__}",
           f"# seed={SEED}  n_boot={N_BOOT}  fimo_cut={FIMO_CUT}",
           f"# command: {' '.join(sys.argv)}", ""]
    raw_ps = []
    for tf in TFS:
        bundle = load_bundle(tf)
        labels = bundle["labels"]
        sites = bundle["sites"]
        all_ids = sites["site_id"].to_numpy()
        in_space_ids = np.array(sorted(bundle["site2wins"].keys()))
        K = len(in_space_ids)

        oof_score, oof_label, oof_ci, oof_pos, oof_widx = align_oof(bundle, tf)
        fim = np.load(EVAL_RESULTS / f"fimo_scores_{tf}.npz")
        fscore = fim["score"].astype(np.float64)
        assert (fim["label"] == labels).all(), "label mismatch vs cache"

        # ---------------- matched count at FIMO's default N ----------------
        N = int(np.sum(fscore >= FIMO_CUT))
        fimo_mask = fscore >= FIMO_CUT
        ds_full = _oof_scores_full(oof_score, oof_widx, len(labels))
        ds_mask = topn_mask(ds_full, N)
        ds_stats, ds_rec = method_stats(ds_mask, bundle, in_space_ids, N)
        fi_stats, fi_rec = method_stats(fimo_mask, bundle, in_space_ids, N)
        ps = paired_stats(ds_rec, fi_rec)
        raw_ps.append(ps["mcnemar_exact_two_sided_p"])
        ds_stats["N_predictions"] = N
        fi_stats["N_predictions"] = N

        # verification vs reviewer's table and stored json
        stored = json.load(open(EVAL_RESULTS / f"fimo_fair_{tf}.json"))
        sm = stored["matched_count"]
        rev = REVIEWER[tf]
        repro[tf] = dict(
            N=N,
            recomputed=dict(dsites_recovered_sites=ds_stats["recovered_sites"],
                            fimo_recovered_sites=fi_stats["recovered_sites"],
                            dsites_top_n=int(N)),
            reviewer_table=dict(dsites_recovered_sites=rev[0],
                                fimo_recovered_sites=rev[1], N=rev[2]),
            stored_json=dict(dsites_recovered_sites=sm["dsites"]["recovered_sites"],
                             fimo_recovered_sites=sm["fimo"]["recovered_sites"],
                             N=sm["N"]),
            matches_reviewer=(ds_stats["recovered_sites"] == rev[0]
                              and fi_stats["recovered_sites"] == rev[1]
                              and N == rev[2]),
            matches_stored_json=(ds_stats["recovered_sites"] == sm["dsites"]["recovered_sites"]
                                 and fi_stats["recovered_sites"] == sm["fimo"]["recovered_sites"]
                                 and N == sm["N"]),
            pooled_oof_windows=int(len(oof_score)),
            universe_windows=int(len(labels)))

        paired[tf] = dict(N=N, dsites=ds_stats, fimo=fi_stats, paired=ps,
                          n_sites_total=int(len(all_ids)),
                          n_sites_in_space=K)

        # ---------------- reverse top-5% comparison ----------------------
        N2 = max(1, int(0.05 * len(oof_score)))
        ds2_mask = topn_mask(ds_full, N2)
        s2 = np.where(fscore > 0, fscore, -np.inf)
        top2 = np.argsort(-s2, kind="stable")[:N2]
        fi2_mask = np.zeros(len(labels), dtype=bool)
        fi2_mask[top2] = True
        fi2_mask &= fscore > 0          # unreported windows carry score 0
        d2, r2d = method_stats(ds2_mask, bundle, in_space_ids, N2)
        f2, r2f = method_stats(fi2_mask, bundle, in_space_ids, N2)
        reverse_rows[tf] = dict(
            N=N2,
            dsites=dict(recovered_sites=d2["recovered_sites"],
                        recall=d2["recall"], site_precision=d2["site_precision_at_N"]),
            fimo=dict(recovered_sites=f2["recovered_sites"], recall=f2["recall"],
                      site_precision=f2["site_precision_at_N"]),
            paired=paired_stats(r2d, r2f))

        log.append(f"[{tf}] N={N}  DS {ds_stats['recovered_sites']}/{K} "
                   f"FIMO {fi_stats['recovered_sites']}/{K}  b={ps['b_ds_only']} "
                   f"c={ps['c_fimo_only']} p={ps['mcnemar_exact_two_sided_p']}")
        print(log[-1], flush=True)

    # Holm correction across the four factors (step-up):
    ps_arr = np.array([p if p is not None else 1.0 for p in raw_ps])
    order = np.argsort(ps_arr)
    m = len(ps_arr)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * ps_arr[idx])
        adj[idx] = min(running, 1.0)
    for i, tf in enumerate(TFS):
        paired[tf]["paired"]["mcnemar_p_holm"] = float(adj[i])

    # ---------------- outputs ----------------
    repro_out = dict(protocol=__doc__, seed=SEED, factors=repro)
    with open(OUT_RES / "matched_counts_reproduction.json", "w") as fh:
        json.dump(repro_out, fh, indent=2, default=str)

    paired_out = dict(protocol=__doc__, seed=SEED, n_bootstrap=N_BOOT,
                      factors=paired, reverse_top5pct=reverse_rows,
                      runtime_s=round(time.time() - t0, 1))
    with open(OUT_RES / "paired_site_recovery.json", "w") as fh:
        json.dump(paired_out, fh, indent=2, default=str)

    rows = []
    for tf in TFS:
        p = paired[tf]
        for method in ("dsites", "fimo"):
            s = p[method]
            rows.append(dict(
                TF=tf, method="D-Sites" if method == "dsites" else "FIMO",
                N=s["N_predictions"],
                sites_recovered=s["recovered_sites"],
                in_space_sites=s["in_space_sites"],
                recall=s["recall"], recall_ci95_lo=s["recall_wilson95"][0],
                recall_ci95_hi=s["recall_wilson95"][1],
                site_precision_at_N=s["site_precision_at_N"],
                precision_ci95_lo=s["precision_wilson95"][0],
                precision_ci95_hi=s["precision_wilson95"][1],
                b_ds_only=p["paired"]["b_ds_only"] if method == "dsites" else "",
                c_fimo_only=p["paired"]["c_fimo_only"] if method == "dsites" else "",
                mcnemar_exact_p=p["paired"]["mcnemar_exact_two_sided_p"] if method == "dsites" else "",
                mcnemar_p_holm=p["paired"]["mcnemar_p_holm"] if method == "dsites" else "",
                OR_b_over_c=p["paired"]["OR_b_over_c"] if method == "dsites" else "",
                OR_ci95=p["paired"]["OR_ci95"] if method == "dsites" else "",
                boot_delta_sites_ci95=p["paired"]["bootstrap_delta_sites_ci95"] if method == "dsites" else "",
                boot_delta_recall_ci95=p["paired"]["bootstrap_delta_recall_ci95"] if method == "dsites" else ""))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "table2_matched_counts.csv", index=False)

    log.append("")
    log.append(f"runtime {time.time()-t0:.1f}s")
    with open(OUT_LOG / "major1.log", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("outputs written:", OUT_RES, OUT_TAB, OUT_LOG, flush=True)


def _oof_scores_full(oof_score, oof_widx, n):
    """Pooled-OOF scores placed into a full universe-length array
    (-inf where not pooled)."""
    s = np.full(n, -np.inf)
    s[oof_widx] = oof_score
    return s


if __name__ == "__main__":
    main()
