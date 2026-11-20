#!/usr/bin/env python3
"""
fnr_operon_matched_counts.py

Rebuilds the FNR operon-level analysis on ONE consistent denominator with a
PWM-only transfer comparison and a matched-count method comparison.

READ-ONLY inputs (evaluation/ is never modified):
  evaluation/results/fnr_transfer_predictions.csv   hybrid full OOB set (39,482)
  evaluation/results/fnr_lt2_window_scores.csv.gz   all LT2 window probabilities
  evaluation/results/fimo_fnr_lt2_promoters.tsv     FIMO LT2 matches (819 at p<1e-4)
  evaluation/data/LT2_operons_heuristic_gap50.tsv   2,750 operons
  evaluation/data/behera2020_FNR_DEPs_LT2mapped.csv 124 unique LT2 DEP genes
  evaluation/results/cache/{FNR_ecoli,FNR_lt2}_windows.pkl

PART 1 — ONE DENOMINATOR. Enrichment-vs-retained curve with x = fraction of
the FULL hybrid OOB set (39,482), grid {0.1%, 0.25%, 0.5%, 1%, 2%, 5%, 10%,
25%, 50%, 100%}. Per point: n, background operon hit rate, DEP-all/activated/
repressed hit rates, fold enrichment vs background, one-sided Fisher p
(greater), 95% CI for the fold (log risk-ratio method). This JSON/CSV is the
single source for revised Table 3 and Figure 5B.

PART 2 — MATCHED-COUNT METHOD COMPARISON at N=819 (FIMO's p<1e-4 count):
  (a) per-method background-corrected Fisher tests (all/activated/repressed)
      with fold CIs;
  (b) PAIRED operon-level comparison (both methods scored on the same 2,750
      operons; operon hit = >=1 prediction in its promoter window):
      discordant operon counts, exact McNemar (scipy binomtest, two-sided),
      Holm correction across the three operon groups per comparison, paired
      cluster bootstrap (10,000 resamples, seed 42, operons as units) for the
      hit-rate difference. Comparisons: hybrid vs FIMO, hybrid vs PWM-only,
      PWM-only vs FIMO. Dependence caveat (adjacent operons share promoter
      space) is stated in the checkpoint.

PART 3 — PWM-ONLY TRANSFER MODEL (leakage-proof; isolates shape features):
  RandomForest(n_estimators=300, seed=42, oob_score=True) on the SAME 97
  RegulonDB E. coli sites with the 3 non-shape features (PWM log-odds, GC,
  Shannon entropy; feature_idx [0,1,2] of the standard vector):
  - positives: best 14-mer within +/-10 bp of each site, both strands
    (predictor's exact extraction);
  - negatives: E. coli promoter windows excluding +/-10 bp around ALL 97
    sites, neg_ratio 5, rng seed 42 (both strand orientations);
  - validation: grouped 5-fold CV via cross_validation.run_cv
    (feature_idx=[0,1,2]; leakage assertion inside; writes only to the R2
    workspace) -> per-fold + pooled AP, OOB accuracy, OOB-F1 cutoff;
  - transfer: scan LT2 promoter windows (both contigs) -> probabilities;
    full set (>= own OOB-F1 cutoff) and top-819 sets;
  - operon tests at 819 for the PWM-only model (same machinery).

No retraining of the hybrid model; deterministic (seed 42). Outputs under
the R2 workspace only.
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
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]   # repository root
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(ROOT / "Software" / "D-Sites" / "src"))
sys.path.insert(0, str(EVAL / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))     # r2_major1 helpers

import eval_common as ec
from dsites import predictor as dp
from cross_validation import run_cv
from fnr_operon_enrichment import (load_operons, load_deps, map_deps_to_operons,
                                   operon_promoters, overlap_mask, group_test)
from r2_major1_matched_pairs import wilson

OUT = EVAL / "results" / "fnr_operon"
TAB = EVAL / "tables"
FIG = EVAL / "figures_local"
LOGD = EVAL / "logs"
PWM_DIR = OUT / "pwm_only_model"
for d in (OUT, TAB, FIG, LOGD, PWM_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 10_000
GRID = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
MATCH_N = 819          # FIMO p<1e-4 count on LT2
N_TREES = 300

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


def holm(pvals):
    """pvals: list of (name, p|None). Returns {name: adj_p}."""
    valid = [(n, p) for n, p in pvals if p is not None]
    order = sorted(valid, key=lambda t: t[1])
    m = len(order)
    running, out = 0.0, {}
    for rank, (n, p) in enumerate(order):
        running = max(running, (m - rank) * p)
        out[n] = min(running, 1.0)
    for n, p in pvals:
        if p is None:
            out[n] = None
    return out


def paired_operon(a_hit, b_hit, n_boot=N_BOOT, seed=SEED):
    """Paired operon comparison: a_hit, b_hit boolean arrays over the same
    operons. Returns discordants, exact McNemar p, bootstrap CI for the
    hit-rate difference a-b (operons as resampling units)."""
    a_hit = np.asarray(a_hit, dtype=bool)
    b_hit = np.asarray(b_hit, dtype=bool)
    K = len(a_hit)
    b_only = int(np.sum(a_hit & ~b_hit))
    c_only = int(np.sum(~a_hit & b_hit))
    p = float(binomtest(b_only, b_only + c_only, 0.5).pvalue) \
        if b_only + c_only > 0 else None
    rng = np.random.default_rng(seed)
    diff = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, K, K)
        diff[i] = a_hit[idx].mean() - b_hit[idx].mean()
    return dict(K=K, a_only=b_only, b_only=c_only, mcnemar_exact_p=p,
                boot_hitrate_diff_ci95=[float(np.percentile(diff, 2.5)),
                                        float(np.percentile(diff, 97.5))])


def group_masks(op):
    return {"DEP_all": (op["n_dep_genes"] > 0).to_numpy(),
            "DEP_activated": (op["n_dep_activated"] > 0).to_numpy(),
            "DEP_repressed": (op["n_dep_repressed"] > 0).to_numpy()}


def fisher_block(hit_mask, op):
    return {g: group_test(hit_mask, m, g) for g, m in group_masks(op).items()}


# ==========================================================================
# Part 3 first (PWM-only model) so Part 2 can use it
# ==========================================================================
def pwm_only_transfer():
    log("[PWM-only] grouped 5-fold CV on 97 E. coli FNR sites (3 features)...")
    t0 = time.time()
    cv = run_cv("FNR_ecoli", n_splits=5, n_trees=N_TREES,
                feature_idx=[0, 1, 2], n_jobs=6, outdir=PWM_DIR,
                tag="_pwm_only", save_oof=False, write_json=True, verbose=False)
    assert cv["leakage_assertion"] == "PASSED", "LEAKAGE in PWM-only CV"
    log(f"[PWM-only] CV done ({time.time()-t0:.0f}s): "
        f"pooled_AP={cv['pooled_PR_AUC']:.4f} OOB acc (per-fold mean)="
        f"{np.mean([f['oob_score'] for f in cv['per_fold']]):.3f}")

    # final model on ALL 97 sites (identical protocol to fnr_transfer.py,
    # feature subset [0,1,2])
    b_ec = ec.build_bundle("FNR_ecoli")
    b_lt2 = ec.build_bundle("FNR_lt2")
    sites = b_ec["sites"]
    pwm, k, _ = ec.load_motif_pwm(b_ec["cfg"]["motif"])
    Xpos = ec.kmer_features(sites["kmer"].tolist(), pwm, b_ec["bg"])[:, [0, 1, 2]]
    rng = np.random.default_rng(SEED)
    eligible = np.nonzero(b_ec["labels"] == 0)[0]
    n_neg = ec.NEG_RATIO * len(sites)
    sel = eligible[rng.choice(len(eligible), size=n_neg, replace=False)]
    neg_rows = np.sort(np.concatenate([2 * sel, 2 * sel + 1]))
    X_tr = np.vstack([Xpos, b_ec["X"][neg_rows][:, [0, 1, 2]]]).astype(np.float32)
    y_tr = np.concatenate([np.ones(len(sites), int),
                           np.zeros(len(neg_rows), int)])
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=6,
                                 random_state=SEED, oob_score=True)
    clf.fit(X_tr, y_tr)
    oob = clf.oob_decision_function_[:, 1]
    ok = ~np.isnan(oob)
    cutoff, oob_f1 = ec.f1_cutoff(y_tr[ok], oob[ok])
    log(f"[PWM-only] final model: OOB acc={clf.oob_score_:.3f}, "
        f"OOB-F1 cutoff={cutoff:.4f}")

    # transfer scan of LT2
    t0 = time.time()
    X3 = b_lt2["X"][:, [0, 1, 2]]
    p2 = clf.predict_proba(X3)[:, 1].astype(np.float32)
    prob = np.maximum(p2[0::2], p2[1::2])
    win_ci, win_pos = b_lt2["win_ci"], b_lt2["win_pos"]
    contigs = b_lt2["contigs"]
    df = pd.DataFrame({
        "contig": np.array(contigs, dtype=object)[win_ci],
        "start": win_pos, "end": win_pos + k - 1,
        "strand": np.where(p2[1::2] > p2[0::2], "-", "+"),
        "score": prob})
    df = df.sort_values("score", ascending=False, kind="stable")
    df.to_csv(PWM_DIR / "fnr_lt2_window_scores_pwm.csv.gz", index=False,
              compression="gzip")
    full = df[df["score"] >= cutoff]
    full.to_csv(PWM_DIR / "fnr_transfer_predictions_pwm.csv", index=False)
    top819 = df.head(MATCH_N)
    top819.to_csv(PWM_DIR / "fnr_transfer_predictions_top819_pwm.csv",
                  index=False)
    log(f"[PWM-only] LT2 scan ({time.time()-t0:.0f}s): {len(df)} windows, "
        f"{len(full)} full-set predictions, top819 saved")

    runinfo = dict(
        protocol="PWM-only (features [PWM_logodds, GC, Entropy]; no shape) "
                 "transfer, identical protocol to fnr_transfer.py",
        n_pos=int(len(sites)), n_neg_rows=int(len(neg_rows)),
        n_trees=N_TREES, seed=SEED, n_jobs=6,
        cutoff_oob_f1=float(cutoff), oob_f1=float(oob_f1),
        oob_score=float(clf.oob_score_),
        cv_json=str(PWM_DIR / "cv_FNR_ecoli_pwm_only.json"),
        n_lt2_windows_scored=int(len(df)),
        n_full_set=int(len(full)),
        n_top819=int(len(top819)),
        versions=ec.runinfo(),
        outputs=[str(p) for p in (
            PWM_DIR / "cv_FNR_ecoli_pwm_only.json",
            PWM_DIR / "fnr_transfer_predictions_pwm.csv",
            PWM_DIR / "fnr_transfer_predictions_top819_pwm.csv",
            PWM_DIR / "fnr_lt2_window_scores_pwm.csv.gz")])
    with open(PWM_DIR / "fnr_pwm_only_runinfo.json", "w") as fh:
        json.dump(runinfo, fh, indent=2, default=str)
    return df


# ==========================================================================
# Parts 1+2 driver
# ==========================================================================
def main():
    t0 = time.time()
    log(f"# r2_major2_fnr_operon.py — started {time.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
        f"scipy {__import__('scipy').__version__}  dsites {dp.__version__}"
        f"  seed={SEED}")

    # ---------------- shared operon scaffolding ----------------
    op = load_operons()
    dep, n_rows = load_deps()
    op, dep, map_report = map_deps_to_operons(dep, op)
    b_lt2 = ec.build_bundle("FNR_lt2")
    contig_lens = {c: len(s) for c, s in b_lt2["contig_seqs"].items()}
    oprom = operon_promoters(op, contig_lens)
    gmask = group_masks(op)
    N_OPERONS = len(op)
    N_DEP = int(gmask["DEP_all"].sum())
    N_ACT = int(gmask["DEP_activated"].sum())
    N_REP = int(gmask["DEP_repressed"].sum())
    log(f"[setup] {N_OPERONS} operons; DEP {N_DEP} (act {N_ACT}, rep {N_REP}); "
        f"124/124 DEP genes mapped")

    # ---------------- prediction sets ----------------
    full_h = pd.read_csv(EVAL / "results" / "fnr_transfer_predictions.csv")
    full_h = full_h.sort_values("score", ascending=False, kind="stable")
    N_full = len(full_h)
    assert N_full == 39482, f"unexpected full-set size {N_full}"

    fimo = ec.parse_fimo_tsv(EVAL / "results" / "fimo_fnr_lt2_promoters.tsv")
    fimo = fimo.rename(columns={"gstart": "s", "gend": "e"})
    fimo819 = fimo[fimo["pvalue"] < 1e-4]
    assert len(fimo819) == MATCH_N, f"FIMO p<1e-4 count {len(fimo819)} != 819"
    fimo_mask = overlap_mask(oprom, fimo819[["contig", "s", "e"]])

    log("[PWM-only] training + transfer ...")
    pwm_scores = pwm_only_transfer()      # sorted desc LT2 window df
    pwm819 = pwm_scores.head(MATCH_N)
    pwm_mask = overlap_mask(oprom, pwm819[["contig", "start", "end"]]
                            .rename(columns={"start": "s", "end": "e"}))
    pwm_full_mask = overlap_mask(oprom, pwm_scores[pwm_scores["score"] >=
        json.load(open(PWM_DIR/"fnr_pwm_only_runinfo.json"))["cutoff_oob_f1"]]
        [["contig", "start", "end"]].rename(columns={"start": "s", "end": "e"}))

    # ---------------- Part 1: unified curve ----------------
    log("[part1] unified curve over fraction of the full set ...")
    curve = []
    for frac in GRID:
        n = N_full if frac == 1.0 else max(1, int(frac * N_full))
        sub = full_h.head(n)
        hits = overlap_mask(oprom, sub[["contig", "start", "end"]]
                            .rename(columns={"start": "s", "end": "e"}))
        row = dict(fraction_of_full_set=frac, n_predictions=int(n),
                   n_operons_hit=int(hits.sum()),
                   background_hit_rate=float(hits.mean()))
        for g, m in gmask.items():
            t = group_test(hits, m, g)
            row[f"{g}_hits"] = t["group_hits"]
            row[f"{g}_rate"] = t["group_rate"]
            row[f"{g}_fold"] = t["fold_enrichment"]
            row[f"{g}_fold_ci95"] = t["fold_ci95"]
            row[f"{g}_fisher_p"] = t["fisher_p_greater"]
        curve.append(row)
        log(f"  top {frac:6.3f} (n={n:6d}): bg={row['background_hit_rate']:.3f} "
            f"DEP {row['DEP_all_hits']}/{N_DEP} fold={row['DEP_all_fold']:.2f} "
            f"p={row['DEP_all_fisher_p']:.2e}")
    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(OUT / "fnr_enrichment_curve_fullset_fraction.csv",
                    index=False)
    with open(OUT / "fnr_enrichment_curve_fullset_fraction.json", "w") as fh:
        json.dump(dict(protocol=__doc__, seed=SEED, N_full_set=N_full,
                       grid=GRID, dep_mapping=map_report, curve=curve), fh,
                  indent=2, default=str)

    # verification vs evaluation/results/fnarest_nested_subset_enrichment.json
    nested = json.load(open(EVAL / "results" / "fnr_nested_subset_enrichment.json"))
    def find_row(n):
        for r in curve:
            if r["n_predictions"] == n:
                return r
    ver = {}
    for n, key in ((39, "dsites_full_nested0.1pct"),
                   (1974, "dsites_full_nested5pct")):
        mine, ref = find_row(n), nested[key]
        ver[key] = dict(
            n=n,
            recomputed=dict(bg=mine["background_hit_rate"],
                            dep_all_hits=mine["DEP_all_hits"],
                            dep_all_fold=mine["DEP_all_fold"],
                            act_hits=mine["DEP_activated_hits"],
                            act_fold=mine["DEP_activated_fold"]),
            nested_json=dict(bg=ref["DEP_all"]["bg_rate"],
                             dep_all_hits=ref["DEP_all"]["hits"],
                             dep_all_fold=ref["DEP_all"]["fold"],
                             act_hits=ref["DEP_activated"]["hits"],
                             act_fold=ref["DEP_activated"]["fold"]),
            consistent=(abs(mine["background_hit_rate"] - ref["DEP_all"]["bg_rate"]) < 1e-12
                        and mine["DEP_all_hits"] == ref["DEP_all"]["hits"]
                        and mine["DEP_activated_hits"] == ref["DEP_activated"]["hits"]))
    for k_, v_ in ver.items():
        log(f"[verify] {k_}: consistent={v_['consistent']}")

    # ---------------- Part 2: matched-819 ----------------
    log("[part2] matched-count method comparison at N=819 ...")
    hyb819 = full_h.head(MATCH_N)
    hyb_mask = overlap_mask(oprom, hyb819[["contig", "start", "end"]]
                            .rename(columns={"start": "s", "end": "e"}))

    methods = {"hybrid": hyb_mask, "fimo": fimo_mask, "pwm_only": pwm_mask}
    fisher = {mth: fisher_block(msk, op) for mth, msk in methods.items()}

    comparisons = {}
    for a_name, b_name in (("hybrid", "fimo"), ("hybrid", "pwm_only"),
                           ("pwm_only", "fimo")):
        a, b = methods[a_name], methods[b_name]
        per_group = {}
        for g, m in gmask.items():
            per_group[g] = paired_operon(a[m], b[m])
        adj = holm([(g, per_group[g]["mcnemar_exact_p"]) for g in per_group])
        for g in per_group:
            per_group[g]["mcnemar_p_holm"] = adj[g]
        comparisons[f"{a_name}_vs_{b_name}"] = per_group

    matched = dict(
        N=MATCH_N, n_operons=N_OPERONS,
        n_hits={m: int(msk.sum()) for m, msk in methods.items()},
        background_rate={m: float(msk.mean()) for m, msk in methods.items()},
        fisher_per_method=fisher, paired=comparisons,
        hybrid_full_set_n=N_full,
        pwm_full_set_n=int(json.load(open(PWM_DIR / "fnr_pwm_only_runinfo.json"))["n_full_set"]),
        verification_vs_nested=ver)
    with open(OUT / "fnr_operon_matched819.json", "w") as fh:
        json.dump(matched, fh, indent=2, default=str)

    # PWM-only full-set operon stats (for the transfer/ dir)
    pwm_operon = fisher_block(pwm_full_mask, op)
    with open(PWM_DIR / "operon_stats_pwm.json", "w") as fh:
        json.dump(dict(n_full_set=matched["pwm_full_set_n"],
                       background_hit_rate=float(pwm_full_mask.mean()),
                       groups=pwm_operon), fh, indent=2, default=str)

    for cmp_, groups in comparisons.items():
        ga = groups["DEP_all"]
        log(f"[paired] {cmp_}: DEP-all b={ga['a_only']} c={ga['b_only']} "
            f"p={ga['mcnemar_exact_p']:.3g} (holm {ga['mcnemar_p_holm']:.3g}) "
            f"boot dRate CI={ga['boot_hitrate_diff_ci95']}")

    # ---------------- table 3 + figure ----------------
    make_table3(curve_df, matched)
    make_figure(curve_df, matched)

    log(f"runtime {time.time()-t0:.0f}s")
    with open(LOGD / "fnr_operon_matched_counts.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print("outputs written", flush=True)


def make_table3(curve_df, matched):
    rows = []
    for _, r in curve_df.iterrows():
        rows.append(dict(section="curve (fraction of full OOB set)",
                         fraction_of_full=r["fraction_of_full_set"],
                         n_predictions=r["n_predictions"],
                         background_rate=r["background_hit_rate"],
                         DEP_all=f"{r['DEP_all_hits']}/109",
                         DEP_all_rate=r["DEP_all_rate"],
                         DEP_all_fold=r["DEP_all_fold"],
                         DEP_all_fisher_p=r["DEP_all_fisher_p"],
                         activated=f"{r['DEP_activated_hits']}/49",
                         activated_rate=r["DEP_activated_rate"],
                         activated_fold=r["DEP_activated_fold"],
                         repressed=f"{r['DEP_repressed_hits']}/60",
                         repressed_rate=r["DEP_repressed_rate"],
                         repressed_fold=r["DEP_repressed_fold"]))
    for mth in ("fimo", "hybrid", "pwm_only"):
        f = matched["fisher_per_method"][mth]
        rows.append(dict(section="matched-819", fraction_of_full="",
                         n_predictions=819 if mth != "hybrid" else 819,
                         background_rate=matched["background_rate"][mth],
                         DEP_all=f"{f['DEP_all']['group_hits']}/109",
                         DEP_all_rate=f["DEP_all"]["group_rate"],
                         DEP_all_fold=f["DEP_all"]["fold_enrichment"],
                         DEP_all_fisher_p=f["DEP_all"]["fisher_p_greater"],
                         activated=f"{f['DEP_activated']['group_hits']}/49",
                         activated_rate=f["DEP_activated"]["group_rate"],
                         activated_fold=f["DEP_activated"]["fold_enrichment"],
                         repressed=f"{f['DEP_repressed']['group_hits']}/60",
                         repressed_rate=f["DEP_repressed"]["group_rate"],
                         repressed_fold=f["DEP_repressed"]["fold_enrichment"]))
    pd.DataFrame(rows).to_csv(TAB / "table3_fnr_operon.csv", index=False)


def make_figure(curve_df, matched):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ---- panel A: bars ----
    ax = axes[0]
    methods = [("fimo", "FIMO p<1e-4"), ("hybrid", "D-Sites hybrid"),
               ("pwm_only", "D-Sites PWM-only")]
    cats = [("background_rate", "background", "#9e9e9e"),
            ("DEP_all_rate", "DEP all", "#386cb0"),
            ("DEP_activated_rate", "FNR-activated", "#7fc97f")]
    x = np.arange(len(methods))
    w = 0.25
    for ci, (key, lab, col) in enumerate(cats):
        vals = [matched["background_rate"][m] for m, _ in methods] if key == "background_rate" \
            else [matched["fisher_per_method"][m][{"DEP_all_rate": "DEP_all",
                                                  "DEP_activated_rate": "DEP_activated"}[key]]["group_rate"]
                  for m, _ in methods]
        ax.bar(x + (ci - 1) * w, vals, w, label=lab, color=col)
    for xi, (m, _) in enumerate(methods):
        p = matched["fisher_per_method"][m]["DEP_all"]["fisher_p_greater"]
        ax.text(xi + 0 * w, matched["fisher_per_method"][m]["DEP_all"]["group_rate"] + 0.012,
                f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}", ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([mlab for _, mlab in methods], fontsize=9)
    ax.set_ylabel("fraction of operons with a hit in the promoter window")
    ax.set_ylim(bottom=0)
    ax.set_title("A. Matched-count comparison (N=819 predictions)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # ---- panel B: unified curve ----
    ax = axes[1]
    ax2 = ax.twinx()
    x = curve_df["fraction_of_full_set"] * 100
    for col, lab, c in (("DEP_all_fold", "DEP all", "#386cb0"),
                        ("DEP_activated_fold", "FNR-activated", "#7fc97f"),
                        ("DEP_repressed_fold", "FNR-repressed", "#fdc086")):
        ax.plot(x, curve_df[col], "o-", color=c, label=f"fold ({lab})")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.plot(x, -np.log10(curve_df["DEP_all_fisher_p"]), "x--", color="#d62728",
             label="-log10 Fisher p (DEP all)")
    ax2.axhline(-np.log10(0.05), color="#d62728", ls=":", lw=1)
    # exploratory n=39 marker
    n39 = curve_df.iloc[0]
    ax.annotate(f"n=39 (0.1% of full):\nexploratory\n(2/49 activated hits)",
                xy=(n39["fraction_of_full_set"] * 100, n39["DEP_all_fold"]),
                xytext=(1.2, 2.1), fontsize=8, va="top",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}%" for v in x], fontsize=8)
    ax.set_xlabel("fraction of the full OOB prediction set retained (39,482)")
    ax.set_ylabel("fold enrichment of DEP operons vs background")
    ax2.set_ylabel("-log10(Fisher p, one-sided greater)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("B. Unified enrichment curve (single denominator)")
    fig.suptitle("FNR transfer (E. coli-trained) — operon-level analysis, matched-count methods "
                 "FNR operon-level validation", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_fnr_operon_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
