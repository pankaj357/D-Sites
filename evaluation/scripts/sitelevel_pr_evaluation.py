#!/usr/bin/env python3
"""
sitelevel_pr_evaluation.py

Site-level precision-recall as the PRIMARY evaluation, on the IDENTICAL
window universe and IDENTICAL positive-site definitions for both tools, per
factor (AmrZ, GlxR, CodY, FNR_ecoli).

FIMO score-floor decision (documented, Task B.1):
  The stored fimo_scores_{TF}.npz arrays (evaluation/scripts/run_fimo_fair.py
  -> eval_common.fimo_scores_to_windows) fill windows FIMO did not report
  with score 0. Our FIMO command was `fimo --thresh 1e-2 --text ...`, so an
  unreported window has p > 1e-2 -- nothing stronger. Score 0 corresponds to
  p = 1, a statement FIMO never made; it also creates the artificial
  straight segment in Fig. 4A (a mass of tied 'p=1' windows). The defensible
  convention is a SCORE FLOOR AT THE REPORTING THRESHOLD:
  unreported windows get -log10(1e-2) = 2.0. Two variants are implemented:
    (a) floor-at-threshold  -- PRIMARY (both per-window and per-site);
    (b) reported-windows-only -- sensitivity (rank only sites with >=1
        reported window; per method).
  No other score invention is used anywhere.

Site-level PR definition (PRIMARY):
  - Site universe = ALL known sites of the factor; positive = IN-SPACE site
    (site interval +/-10 bp intersects any promoter window; the harmonized
    definition). Out-of-space sites are unrecoverable in this search space
    and act as negatives ranked at the bottom (score floor).
  - Site score = max over the site's overlapping windows of the (floored)
    per-window score. D-Sites: pooled out-of-fold probabilities (oof npz)
    placed on the universe; windows absent from pooling (multi-fold
    conflicts) take the minimum pooled probability. FIMO: floored -log10(p).
  - Rank sites by score; cumulative TP = in-space sites seen; precision =
    TP / #sites ranked; AP = average precision over the ranked list;
    no-skill = prevalence of in-space sites (over the ranked list).
  Window-level AP (existing computation) is reported side-by-side for the
  relationship explanation, with ONE unified window prevalence per factor
  (full-universe labels -- the two tools' old ranked lists had different
  denominators, which resolves the previously reported 0.0029-vs-0.0035 AmrZ discrepancy).

VERIFICATION: the old supplementary site-level curves
(evaluation/results/pr_sitelevel_{TF}.csv) are REGENERATED into the R2
workspace with make_pr_curves.sitelevel_curves (imported, unmodified) and
diffed against the stored CSVs; definitional differences vs the new
site-level PR are documented in the checkpoint.

READ-ONLY inputs: evaluation/results/{oof,fimo_scores}_{TF}.npz,
evaluation/results/cache/{TF}_windows.pkl, evaluation/results/pr_curve_{TF}.csv,
evaluation/results/pr_sitelevel_{TF}.csv. Nothing in evaluation/ is modified.
Deterministic (seed 42).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]   # repository root

EVAL = ROOT / "evaluation"
OUT = EVAL / "results" / "sitelevel_pr"
TAB = EVAL / "tables"
FIG = EVAL / "figures_local"
LOGD = EVAL / "logs"
VERIFY = OUT / "verification_previous_definitions"
for d in (OUT, TAB, FIG, LOGD, VERIFY):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Software" / "D-Sites" / "src"))
sys.path.insert(0, str(EVAL / "scripts"))

import eval_common as ec
from dsites import predictor as dp
from make_pr_curves import sitelevel_curves

SEED = 42
TFS = ["AmrZ", "GlxR", "CodY", "FNR_ecoli"]
FLOOR = 2.0            # -log10(1e-2): FIMO reporting threshold
KEYSHIFT = 40

log_lines = []


def log(m):
    print(m, flush=True)
    log_lines.append(m)


def load_bundle(tf):
    with open(EVAL / "results" / "cache" / f"{tf}_windows.pkl", "rb") as fh:
        return pickle_load(fh)


def pickle_load(fh):
    import pickle
    return pickle.load(fh)


def universe_ds_scores(bundle, tf):
    """Full-universe D-Sites score array from pooled OOF (-inf where not
    pooled)."""
    oof = np.load(EVAL / "results" / f"oof_{tf}.npz")
    key_all = (bundle["win_ci"].astype(np.int64) << KEYSHIFT) | \
        bundle["win_pos"].astype(np.int64)
    order = np.argsort(key_all, kind="stable")
    key_oof = (oof["win_ci"].astype(np.int64) << KEYSHIFT) | \
        oof["win_pos"].astype(np.int64)
    pos = np.searchsorted(key_all[order], key_oof)
    widx = order[pos]
    s = np.full(len(bundle["win_pos"]), -np.inf)
    s[widx] = oof["score"].astype(np.float64)
    return s, oof


def site_scores(bundle, win_scores, fill):
    """Per-site max window score; sites without windows/scores get `fill`."""
    s2w = bundle["site2wins"]
    sites = bundle["sites"]
    out = np.empty(len(sites))
    for i, sid in enumerate(sites["site_id"].to_numpy()):
        w = s2w.get(int(sid))
        if w is None:
            out[i] = fill
        else:
            v = win_scores[w]
            m = v.max()
            out[i] = fill if not np.isfinite(m) else m
    return out


def ap_and_curve(y, s):
    ap = float(average_precision_score(y, s))
    prec, rec, _ = precision_recall_curve(y, s)
    return ap, rec, prec


def main():
    t0 = time.time()
    log(f"# r2_major4_sitelevel_pr.py  seed={SEED}  "
        f"python {sys.version.split()[0]}  dsites {dp.__version__}")

    summary_rows = []
    per_tf = {}
    for tf in TFS:
        bundle = load_bundle(tf)
        sites = bundle["sites"]
        n_sites = len(sites)
        y_inspace = np.array([int(s) in bundle["site2wins"]
                              for s in sites["site_id"].to_numpy()])
        prevalence_sites = float(y_inspace.mean())

        # unified window prevalence (full universe labels = one denominator)
        fim_npz = np.load(EVAL / "results" / f"fimo_scores_{tf}.npz")
        labels = fim_npz["label"].astype(np.int8)
        prevalence_windows = float(labels.mean())
        fimo_raw = fim_npz["score"].astype(np.float64)
        fimo_floor = np.where(fimo_raw > 0, fimo_raw, FLOOR)

        ds_full, oof = universe_ds_scores(bundle, tf)
        ds_floor = float(oof["score"].min())

        # ---------- PRIMARY (variant a: floor at threshold) ----------
        ds_site = site_scores(bundle, ds_full, fill=ds_floor)
        fi_site = site_scores(bundle, fimo_floor, fill=FLOOR)
        ap_ds_a, rec_ds, prec_ds = ap_and_curve(y_inspace, ds_site)
        ap_fi_a, rec_fi, prec_fi = ap_and_curve(y_inspace, fi_site)
        base_a = float(y_inspace.mean())

        # ---------- sensitivity (variant b: reported windows only) ----------
        # D-Sites: sites with >=1 pooled-scored window; FIMO: >=1 reported.
        ds_b_mask = np.isfinite(site_scores(bundle, ds_full, fill=-np.inf))
        fi_b_mask = site_scores(bundle, np.where(fimo_raw > 0, fimo_raw, -np.inf),
                                fill=-np.inf) > -np.inf
        ap_ds_b = float(average_precision_score(y_inspace[ds_b_mask],
                                                ds_site[ds_b_mask]))
        ap_fi_b = float(average_precision_score(y_inspace[fi_b_mask],
                                                fi_site[fi_b_mask]))

        # ---------- window-level AP (existing computation) ----------
        ap_win_ds = float(average_precision_score(
            oof["label"].astype(np.int8), oof["score"].astype(np.float64)))
        ap_win_fi_floor0 = float(average_precision_score(labels, fimo_raw))
        ap_win_fi_floored = float(average_precision_score(labels, fimo_floor))

        per_tf[tf] = dict(
            n_sites=int(n_sites), n_in_space=int(y_inspace.sum()),
            prevalence_inspace_sites=prevalence_sites,
            unified_window_prevalence=prevalence_windows,
            variant_a_floor_at_threshold=dict(
                floor_value=FLOOR,
                dsites=dict(site_AP=ap_ds_a,
                            n_ranked=int(n_sites),
                            no_skill=base_a),
                fimo=dict(site_AP=ap_fi_a, n_ranked=int(n_sites),
                          no_skill=base_a),
                curve_points=dict(dsites_recall=rec_ds.tolist(),
                                  dsites_precision=prec_ds.tolist(),
                                  fimo_recall=rec_fi.tolist(),
                                  fimo_precision=prec_fi.tolist())),
            variant_b_reported_only=dict(
                dsites=dict(site_AP=ap_ds_b, n_ranked=int(ds_b_mask.sum())),
                fimo=dict(site_AP=ap_fi_b, n_ranked=int(fi_b_mask.sum()))),
            window_level=dict(
                dsites_pooled_AP=ap_win_ds,
                fimo_AP_floor0_stored_convention=ap_win_fi_floor0,
                fimo_AP_floored=ap_win_fi_floored))
        summary_rows.append(dict(
            TF=tf, n_sites=n_sites, n_in_space=int(y_inspace.sum()),
            site_prevalence=prevalence_sites,
            unified_window_prevalence=prevalence_windows,
            siteAP_dsites_floor=ap_ds_a, siteAP_fimo_floor=ap_fi_a,
            siteAP_dsites_reported=ap_ds_b, siteAP_fimo_reported=ap_fi_b,
            winAP_dsites_pooled=ap_win_ds,
            winAP_fimo_floor0=ap_win_fi_floor0, winAP_fimo_floored=ap_win_fi_floored))
        log(f"[{tf}] site-level AP (floor): D-Sites {ap_ds_a:.4f} vs FIMO "
            f"{ap_fi_a:.4f} (baseline {base_a:.3f}; in-space "
            f"{int(y_inspace.sum())}/{n_sites}) | window AP pooled DS "
            f"{ap_win_ds:.4f} / FIMO(0-fill) {ap_win_fi_floor0:.4f} / "
            f"FIMO(floored) {ap_win_fi_floored:.4f} | window prevalence "
            f"{prevalence_windows:.5f}")
        with open(OUT / f"sitelevel_pr_{tf}.json", "w") as fh:
            json.dump(per_tf[tf], fh, indent=2, default=str)

    pd.DataFrame(summary_rows).to_csv(OUT / "sitelevel_pr_summary.csv",
                                      index=False)

    # ---------------- verification of old definitions ----------------
    log("[verify] regenerating old pr_sitelevel CSVs (make_pr_curves, "
        "unmodified) and diffing vs stored ...")
    verify_report = {}
    for tf in ["AmrZ", "GlxR", "CodY"]:
        for f in (f"oof_{tf}.npz", f"fimo_scores_{tf}.npz"):
            dst = VERIFY / f
            if not dst.exists():
                dst.symlink_to(EVAL / "results" / f)
        sitelevel_curves(tf, n_boot=1000, seed=SEED, outdir=VERIFY,
                         figdir=VERIFY, verbose=False)
        new = pd.read_csv(VERIFY / f"pr_sitelevel_{tf}.csv")
        old = pd.read_csv(EVAL / "results" / f"pr_sitelevel_{tf}.csv")
        same = new.shape == old.shape and np.allclose(
            new.select_dtypes("number").to_numpy(),
            old.select_dtypes("number").to_numpy(), atol=1e-10, equal_nan=True)
        verify_report[tf] = dict(identical=bool(same), rows_new=len(new),
                                 rows_old=len(old))
        log(f"[verify] {tf}: old-definition regeneration identical = {same}")

    make_figures(per_tf)
    with open(OUT / "sitelevel_pr_verification_old_definitions.json", "w") as fh:
        json.dump(verify_report, fh, indent=2)

    log(f"runtime {time.time()-t0:.0f}s")
    with open(LOGD / "major4.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print("outputs written", flush=True)


def make_figures(per_tf):
    # ---------- PRIMARY: site-level PR ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, tf, panel in zip(axes.ravel(), TFS, "ABCD"):
        d = per_tf[tf]["variant_a_floor_at_threshold"]
        base = d["dsites"]["no_skill"]
        rec_d = np.array(d["curve_points"]["dsites_recall"])
        prc_d = np.array(d["curve_points"]["dsites_precision"])
        rec_f = np.array(d["curve_points"]["fimo_recall"])
        prc_f = np.array(d["curve_points"]["fimo_precision"])
        ax.plot(rec_d, prc_d, color="#1f77b4", lw=2,
                label=f"D-Sites (AP={d['dsites']['site_AP']:.3f})")
        ax.plot(rec_f, prc_f, color="#d62728", lw=2,
                label=f"FIMO (AP={d['fimo']['site_AP']:.3f})")
        ax.axhline(base, color="gray", ls="--", lw=1,
                   label=f"no-skill ({base:.3f})")
        # honest zoom: curves plateau at precision 1.0; show the gap to the
        # no-skill line and the terminal drop into the out-of-space block
        ymin = max(0.0, min(base, 0.985) - 0.015)
        ax.set_ylim(ymin, 1.02)
        ax.set_xlim(0, 1)
        ax.set_xlabel("site-level recall (in-space sites ranked)")
        ax.set_ylabel("site-level precision")
        ax.set_title(f"{panel}. {tf} — site-level PR (primary)", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Site-level precision-recall, identical window universe and site "
                 "definitions (FIMO scores floored at the 1e-2 reporting threshold)",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sitelevel_pr_primary.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ---------- SUPPLEMENTARY: window-level (existing curves, zoomed) ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, tf, panel in zip(axes.ravel(), TFS, "ABCD"):
        stored = EVAL / "results" / f"pr_curve_{tf}.csv"
        if stored.exists():
            df = pd.read_csv(stored)
            prev = float(df["prevalence"].iloc[0])
            series = []
            for method, col in (("D-Sites", "#1f77b4"), ("FIMO", "#d62728")):
                sub = df[df["method"] == method]
                ap = float(sub["AP"].iloc[0])
                series.append((sub["recall"].to_numpy(),
                               sub["precision_full"].to_numpy(), col,
                               f"{method} (AP={ap:.4f})"))
        else:
            # no stored curve for this factor (R1 made 3 TFs): compute the
            # identical window-level curves directly from the npz caches
            oof = np.load(EVAL / "results" / f"oof_{tf}.npz")
            fim = np.load(EVAL / "results" / f"fimo_scores_{tf}.npz")
            series = []
            for (y, s, col, lab) in (
                    (oof["label"].astype(np.int8), oof["score"].astype(np.float64),
                     "#1f77b4", "D-Sites"),
                    (fim["label"].astype(np.int8), fim["score"].astype(np.float64),
                     "#d62728", "FIMO")):
                p_, r_, _ = precision_recall_curve(y, s)
                series.append((r_, p_, col,
                               f"{lab} (AP={average_precision_score(y, s):.4f})"))
            prev = float(fim["label"].mean())
        for rec, prc, col, lab in series:
            ax.plot(rec, prc, color=col, lw=1.8, label=lab)
        ax.axhline(prev, color="gray", ls="--", lw=1,
                   label=f"no-skill ({prev:.4f})")
        ymax = max(prev * 3, max(s[1].max() for s in series)) * 1.15
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("window-level recall")
        ax.set_ylabel("window-level precision")
        ax.set_title(f"{panel}. {tf} — window-level PR (supplementary)", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Window-level precision-recall (existing evaluation, zoomed axes; "
                 "FIMO 0-fill convention: unreported windows = score 0)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig_windowlevel_pr_supplementary.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
