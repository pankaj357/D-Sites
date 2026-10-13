#!/usr/bin/env python3
"""
FNR operon-level enrichment analysis

Outputs:
  results/fnr_operon_enrichment.json          all tables, p-values, rates
  results/fnr_operon_enrichment_summary.csv   compact per-set table
  results/fnr_enrichment_vs_threshold.csv     sweep table
  figures/fnr_operon_enrichment.png           bar chart (y from 0, p annotated)
  figures/fnr_enrichment_vs_threshold.png     sweep curves

"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, hypergeom

import eval_common as ec

UP, DOWN = 300, 50
FIMO_DEFAULT_P = 1e-4
SWEEP_FRACS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
               0.10, 0.25, 0.50, 1.00]

DIRECTIONS = {
    "activated": "down_in_mutant_FNR_activated",
    "repressed": "up_in_mutant_FNR_repressed",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_operons():
    op = pd.read_csv(ec.DATA / "LT2_operons_heuristic_gap50.tsv", sep="\t")
    assert len(op) == 2750, f"expected 2,750 operons, got {len(op)}"
    return op


def load_deps():
    dep = pd.read_csv(ec.DATA / "behera2020_FNR_DEPs_LT2mapped.csv")
    dep = dep.dropna(subset=["lt2_locus_tag"])
    n_rows = len(dep)
    dep = dep.drop_duplicates(subset=["lt2_locus_tag"], keep="first")
    return dep, n_rows


def map_deps_to_operons(dep, op):
    """Returns (op with DEP annotations, mapping report)."""
    tag2op = {}
    for r in op.itertuples():
        for tag in str(r.locus_tags).split(";"):
            tag2op[tag] = r.operon_id
    dep = dep.copy()
    dep["operon_id"] = dep["lt2_locus_tag"].map(tag2op)
    unmapped = dep[dep["operon_id"].isna()]
    act_tags = set(dep.loc[dep["direction"] == DIRECTIONS["activated"],
                           "lt2_locus_tag"])
    rep_tags = set(dep.loc[dep["direction"] == DIRECTIONS["repressed"],
                           "lt2_locus_tag"])
    n_act = np.zeros(len(op), dtype=int)
    n_rep = np.zeros(len(op), dtype=int)
    n_dep = np.zeros(len(op), dtype=int)
    for i, r in enumerate(op.itertuples()):
        tags = str(r.locus_tags).split(";")
        n_act[i] = sum(t in act_tags for t in tags)
        n_rep[i] = sum(t in rep_tags for t in tags)
        n_dep[i] = n_act[i] + n_rep[i]
    op = op.copy()
    op["n_dep_genes"] = n_dep
    op["n_dep_activated"] = n_act
    op["n_dep_repressed"] = n_rep
    report = dict(
        n_dep_rows_file=None, n_dep_unique_genes=int(len(dep)),
        n_dep_genes_mapped=int(dep["operon_id"].notna().sum()),
        n_dep_genes_unmapped=int(len(unmapped)),
        unmapped_tags=unmapped["lt2_locus_tag"].tolist(),
        n_dep_operons=int((op["n_dep_genes"] > 0).sum()),
        n_dep_operons_activated=int((op["n_dep_activated"] > 0).sum()),
        n_dep_operons_repressed=int((op["n_dep_repressed"] > 0).sum()),
        n_dep_operons_mixed=int(((op["n_dep_activated"] > 0) &
                                 (op["n_dep_repressed"] > 0)).sum()),
        n_dep_genes_sharing_operon=int(dep.duplicated("operon_id").sum()),
        dep_genes_per_operon_hist={
            str(k): int(v) for k, v in
            op.loc[op["n_dep_genes"] > 0, "n_dep_genes"].value_counts()
            .sort_index().items()},
    )
    return op, dep, report


def operon_promoters(op, contig_lens):
    ps = np.where(op["strand"] == "+", op["start"] - UP, op["end"] - DOWN)
    pe = np.where(op["strand"] == "+", op["start"] + DOWN, op["end"] + UP)
    lens = op["contig"].map(contig_lens).to_numpy()
    ps = np.clip(ps, 1, lens)
    pe = np.clip(pe, 1, lens)
    return pd.DataFrame({"contig": op["contig"], "ps": ps, "pe": pe})


def load_prediction_sets(n_top5):
    preds = {}
    full = pd.read_csv(ec.RESULTS / "fnr_transfer_predictions.csv")
    top5 = pd.read_csv(ec.RESULTS / "fnr_transfer_predictions_top5pct.csv")
    preds["dsites_full"] = (full.rename(columns={"start": "s", "end": "e"}),
                            f"prob >= OOB-F1 cutoff (n={len(full)})")
    preds["dsites_top5pct"] = (top5.rename(columns={"start": "s", "end": "e"}),
                               f"top 5% of scored windows (n={len(top5)})")
    fimo = ec.parse_fimo_tsv(ec.RESULTS / "fimo_fnr_lt2_promoters.tsv")
    fimo = fimo.rename(columns={"gstart": "s", "gend": "e"})
    f_p = fimo[fimo["pvalue"] < FIMO_DEFAULT_P]
    preds["fimo_p1e-4"] = (f_p, f"FIMO p<1e-4 (n={len(f_p)})")
    f_n = fimo.sort_values("pvalue", kind="stable").head(n_top5)
    capped = len(f_n) < n_top5
    preds["fimo_topN"] = (
        f_n, f"FIMO top-N by p-value, N={n_top5} = |top5pct|"
             + (f" -- CAPPED at {len(f_n)} reported matches (p<1e-2)"
                if capped else ""))
    # additional stringent D-Sites sets (see docstring point 4)
    scores = pd.read_csv(ec.RESULTS / "fnr_lt2_window_scores.csv.gz")
    sc = scores["score"].to_numpy()
    for frac, key in ((0.01, "dsites_top1pct"), (0.001, "dsites_top0.1pct")):
        thr = np.quantile(sc, 1.0 - frac)
        sub = scores[scores["score"] >= thr]
        preds[key] = (sub.rename(columns={"start": "s", "end": "e"})
                      [["contig", "s", "e"]],
                      f"ADDED stringent set: top {frac:.1%} of scored "
                      f"windows (n={len(sub)}, prob>={thr:.4f})")
    order = ["dsites_full", "dsites_top5pct", "dsites_top1pct",
             "dsites_top0.1pct", "fimo_p1e-4", "fimo_topN"]
    preds = {k: preds[k] for k in order if k in preds}
    return preds, dict(fimo_topN_capped=bool(capped),
                       fimo_topN_requested=int(n_top5),
                       fimo_topN_effective=int(len(f_n)),
                       fimo_total_reported=int(len(fimo)))


# --------------------------------------------------------------------------
# overlap + statistics
# --------------------------------------------------------------------------
def overlap_mask(windows, preds):
    """Boolean per window row: does any prediction interval overlap
    [ps, pe] on the same contig (1-bp overlap)."""
    mask = np.zeros(len(windows), dtype=bool)
    widx = np.arange(len(windows))
    for contig in windows["contig"].unique():
        wm = (windows["contig"] == contig).to_numpy()
        pr = preds[preds["contig"] == contig]
        if pr.empty:
            continue
        ps = windows.loc[wm, "ps"].to_numpy(np.int64)
        pe = windows.loc[wm, "pe"].to_numpy(np.int64)
        s = pr["s"].to_numpy(np.int64)
        e = pr["e"].to_numpy(np.int64)
        order = np.argsort(s, kind="stable")
        s_s, e_s = s[order], e[order]
        cmax_e = np.maximum.accumulate(e_s)
        idx = np.searchsorted(s_s, pe, side="right")     # preds with s <= pe
        hit = (idx > 0) & (cmax_e[np.clip(idx - 1, 0, None)] >= ps)
        mask[widx[wm]] = hit
    return mask


def group_test(hit_mask, group_mask, label):
    """2x2: rows = in group (y/n), cols = hit (y/n); Fisher 'greater' +
    hypergeometric tail + fold enrichment (risk ratio) + odds ratio CIs."""
    M = len(hit_mask)
    n = int(group_mask.sum())
    N = int(hit_mask.sum())
    a = int((hit_mask & group_mask).sum())
    b = n - a
    c = N - a
    d = M - a - b - c
    assert min(a, b, c, d) >= 0 and a <= n
    bg_rate = N / M
    grp_rate = a / n if n else 0.0
    fold = grp_rate / bg_rate if bg_rate else np.nan
    or_res = fisher_exact([[a, b], [c, d]], alternative="greater")
    p_fisher = float(or_res.pvalue)
    p_hyper = float(hypergeom.sf(a - 1, M, n, N)) if a > 0 else 1.0
    # 95% log-normal CI for fold enrichment (risk ratio)
    if a > 0 and c > 0 and n > 0 and (M - n) > 0:
        se_rr = np.sqrt(1 / a - 1 / n + 1 / c - 1 / (M - n))
        rr_lo, rr_hi = np.exp(np.log(fold) - 1.96 * se_rr), \
            np.exp(np.log(fold) + 1.96 * se_rr)
    else:
        rr_lo = rr_hi = np.nan
    # 95% log-normal CI for the odds ratio (Haldane-Anscombe if a cell is 0)
    aa, bb, cc, dd = (np.array([a, b, c, d], dtype=float)
                      + (0.5 if min(a, b, c, d) == 0 else 0.0))
    or_ = (aa * dd) / (bb * cc) if bb * cc > 0 else np.nan
    se_or = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    or_lo, or_hi = np.exp(np.log(or_) - 1.96 * se_or), \
        np.exp(np.log(or_) + 1.96 * se_or)
    return dict(group=label, group_size=n, group_hits=a,
                table_2x2=[[a, b], [c, d]],
                background_rate=float(bg_rate), group_rate=float(grp_rate),
                fold_enrichment=float(fold),
                fold_ci95=[float(rr_lo), float(rr_hi)],
                odds_ratio=float(or_), or_ci95=[float(or_lo), float(or_hi)],
                fisher_p_greater=p_fisher, hypergeom_sf=p_hyper,
                significant_005=bool(p_fisher < 0.05))


def enrichment_block(hit_mask, op):
    groups = {
        "DEP_all": (op["n_dep_genes"] > 0).to_numpy(),
        "DEP_activated": (op["n_dep_activated"] > 0).to_numpy(),
        "DEP_repressed": (op["n_dep_repressed"] > 0).to_numpy(),
    }
    out = {g: group_test(hit_mask, m, g) for g, m in groups.items()}
    # direct activated-vs-repressed contrast (two-sided)
    act, rep = groups["DEP_activated"], groups["DEP_repressed"]
    tab = [[int((hit_mask & act).sum()), int((~hit_mask & act).sum())],
           [int((hit_mask & rep).sum()), int((~hit_mask & rep).sum())]]
    out["activated_vs_repressed"] = dict(
        table_2x2=tab, fisher_p_twosided=float(fisher_exact(tab).pvalue),
        note="two-sided Fisher test of hit-rate difference between "
             "FNR-activated and FNR-repressed DEP operons")
    return out


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    op = load_operons()
    dep, n_dep_rows = load_deps()
    op, dep, map_report = map_deps_to_operons(dep, op)
    map_report["n_dep_rows_file"] = int(n_dep_rows)
    bundle = ec.build_bundle("FNR_lt2")
    contig_lens = {c: len(s) for c, s in bundle["contig_seqs"].items()}
    oprom = operon_promoters(op, contig_lens)

    n_top5 = sum(1 for _ in open(ec.RESULTS /
                                 "fnr_transfer_predictions_top5pct.csv")) - 1
    pred_sets, set_meta = load_prediction_sets(n_top5)

    # ---------------- operon level ----------------
    operon_results = {}
    for name, (pdf, desc) in pred_sets.items():
        hits = overlap_mask(oprom, pdf[["contig", "s", "e"]])
        assert hits.sum() <= len(op)
        operon_results[name] = dict(
            description=desc, n_predictions=int(len(pdf)),
            n_operons_hit=int(hits.sum()),
            background_hit_rate=float(hits.mean()),
            groups=enrichment_block(hits, op))

    # ---------------- gene level (legacy view) ----------------
    gff = bundle["cfg"]["gff"]
    genes = {}
    with open(gff) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            tag = None
            for a in f[8].split(";"):
                if a.startswith("locus_tag="):
                    tag = a.split("=", 1)[1]
                    break
            if tag and tag not in genes:
                genes[tag] = (f[0], int(f[3]), int(f[4]), f[6])
    gdf = pd.DataFrame([(t, *v) for t, v in genes.items()],
                       columns=["locus_tag", "contig", "start", "end", "strand"])
    gprom = pd.DataFrame({
        "contig": gdf["contig"],
        "ps": np.clip(np.where(gdf["strand"] == "+", gdf["start"] - UP,
                               gdf["end"] - DOWN), 1,
                      gdf["contig"].map(contig_lens).to_numpy()),
        "pe": np.clip(np.where(gdf["strand"] == "+", gdf["start"] + DOWN,
                               gdf["end"] + UP), 1,
                      gdf["contig"].map(contig_lens).to_numpy())})
    dep_act = set(dep.loc[dep["direction"] == DIRECTIONS["activated"],
                          "lt2_locus_tag"])
    dep_rep = set(dep.loc[dep["direction"] == DIRECTIONS["repressed"],
                          "lt2_locus_tag"])
    gdf["is_dep"] = gdf["locus_tag"].isin(dep_act | dep_rep)
    gdf["is_act"] = gdf["locus_tag"].isin(dep_act)
    gdf["is_rep"] = gdf["locus_tag"].isin(dep_rep)
    gene_results = {}
    for name, (pdf, desc) in pred_sets.items():
        hits = overlap_mask(gprom, pdf[["contig", "s", "e"]])
        groups = {"DEP_all": gdf["is_dep"].to_numpy(),
                  "DEP_activated": gdf["is_act"].to_numpy(),
                  "DEP_repressed": gdf["is_rep"].to_numpy()}
        gene_results[name] = dict(
            description=desc, n_predictions=int(len(pdf)),
            n_genes_hit=int(hits.sum()), n_genes_total=int(len(gdf)),
            background_hit_rate=float(hits.mean()),
            groups={g: group_test(hits, m, g) for g, m in groups.items()})

    # ---------------- enrichment vs threshold ----------------
    scores = pd.read_csv(ec.RESULTS / "fnr_lt2_window_scores.csv.gz")
    sweep = []
    for frac in SWEEP_FRACS:
        thr = np.quantile(scores["score"].to_numpy(), 1.0 - frac) \
            if frac < 1.0 else scores["score"].min()
        sub = scores[scores["score"] >= thr]
        hits = overlap_mask(oprom, sub.rename(columns={"start": "s",
                                                       "end": "e"})
                            [["contig", "s", "e"]])
        grp = enrichment_block(hits, op)["DEP_all"]
        act = group_test(hits, (op["n_dep_activated"] > 0).to_numpy(), "a")
        rep = group_test(hits, (op["n_dep_repressed"] > 0).to_numpy(), "r")
        sweep.append(dict(fraction=frac, score_threshold=float(thr),
                          n_predictions=int(len(sub)),
                          n_operons_hit=int(hits.sum()),
                          background_rate=grp["background_rate"],
                          dep_rate=grp["group_rate"],
                          dep_fold=grp["fold_enrichment"],
                          dep_fisher_p=grp["fisher_p_greater"],
                          act_rate=act["group_rate"], act_fold=act["fold_enrichment"],
                          rep_rate=rep["group_rate"], rep_fold=rep["fold_enrichment"]))
        print(f"[sweep] top {frac:5.1%}: thr={thr:.4f} n={len(sub):7d} "
              f"bg={grp['background_rate']:.3f} dep={grp['group_rate']:.3f} "
              f"fold={grp['fold_enrichment']:.2f} p={grp['fisher_p_greater']:.2e}",
              flush=True)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(ec.RESULTS / "fnr_enrichment_vs_threshold.csv", index=False)

    # ---------------- figures ----------------
    make_bar_figure(operon_results)
    make_sweep_figure(sweep_df)

    # ---------------- summary CSV + JSON ----------------
    rows = []
    for level, res in (("operon", operon_results), ("gene", gene_results)):
        for name, r in res.items():
            ga = r["groups"]["DEP_all"]
            rows.append(dict(
                level=level, prediction_set=name,
                n_predictions=r["n_predictions"],
                background_hit_rate=ga["background_rate"],
                dep_hits=ga["group_hits"], dep_total=ga["group_size"],
                dep_hit_rate=ga["group_rate"],
                act_hit_rate=r["groups"]["DEP_activated"]["group_rate"],
                rep_hit_rate=r["groups"]["DEP_repressed"]["group_rate"],
                fold_enrichment=ga["fold_enrichment"],
                fold_ci95_lo=ga["fold_ci95"][0], fold_ci95_hi=ga["fold_ci95"][1],
                odds_ratio=ga["odds_ratio"],
                fisher_p=ga["fisher_p_greater"],
                hypergeom_sf=ga["hypergeom_sf"],
                table_2x2=str(ga["table_2x2"])))
    pd.DataFrame(rows).to_csv(
        ec.RESULTS / "fnr_operon_enrichment_summary.csv", index=False)

    results = dict(
        purpose=__doc__.split("\n")[1],
        dep_mapping=map_report, prediction_set_meta=set_meta,
        n_operons=int(len(op)), n_genes_gene_level=int(len(gdf)),
        operon_level=operon_results, gene_level=gene_results,
        sweep=sweep,
        runinfo=ec.runinfo(dict(wall_seconds=round(time.time() - t0, 1))))
    ec.save_json(results, ec.RESULTS / "fnr_operon_enrichment.json")
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_bar_figure(operon_results):
    sets = list(operon_results.keys())
    labels = {"dsites_full": "D-Sites full", "dsites_top5pct": "D-Sites top 5%",
              "fimo_p1e-4": "FIMO p<1e-4", "fimo_topN": "FIMO top-N",
              "dsites_top1pct": "D-Sites top 1%",
              "dsites_top0.1pct": "D-Sites top 0.1%"}
    cats = ["background", "DEP all", "FNR-activated", "FNR-repressed"]
    colors = ["#9e9e9e", "#386cb0", "#7fc97f", "#fdc086"]
    x = np.arange(len(sets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for ci, (cat, col) in enumerate(zip(cats, colors)):
        vals = []
        for s in sets:
            r = operon_results[s]
            if cat == "background":
                vals.append(r["background_hit_rate"])
            elif cat == "DEP all":
                vals.append(r["groups"]["DEP_all"]["group_rate"])
            elif cat == "FNR-activated":
                vals.append(r["groups"]["DEP_activated"]["group_rate"])
            else:
                vals.append(r["groups"]["DEP_repressed"]["group_rate"])
        ax.bar(x + (ci - 1.5) * width, vals, width, label=cat, color=col)
    for xi, s in enumerate(sets):
        g = operon_results[s]["groups"]["DEP_all"]
        p = g["fisher_p_greater"]
        txt = f"p={p:.1e}" if p >= 0.001 else f"p={p:.0e}"
        ax.text(xi + (1 - 1.5) * width, g["group_rate"] + 0.015, txt,
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[s]}\n(n={operon_results[s]['n_predictions']:,})"
                        for s in sets], fontsize=9)
    ax.set_ylabel("fraction of operons with a hit in the promoter window")
    ax.set_ylim(bottom=0)
    ax.set_title("FNR transfer (E. coli-trained model) — operon-level hit rates on\n"
                 "Salmonella LT2: Behera et al. 2020 DEP operons vs genome-wide background")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ec.FIGURES / "fnr_operon_enrichment.png", dpi=300)
    plt.close(fig)


def make_sweep_figure(sweep_df):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax2 = ax.twinx()
    x = sweep_df["fraction"] * 100
    ax.plot(x, sweep_df["dep_fold"], "o-", color="#386cb0",
            label="fold enrichment (DEP all)")
    ax.plot(x, sweep_df["act_fold"], "s--", color="#7fc97f",
            label="fold enrichment (FNR-activated)")
    ax.plot(x, sweep_df["rep_fold"], "^--", color="#fdc086",
            label="fold enrichment (FNR-repressed)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.plot(x, -np.log10(sweep_df["dep_fisher_p"]), "x-", color="#d62728",
             label="-log10 Fisher p (DEP all)")
    ax2.axhline(-np.log10(0.05), color="#d62728", ls=":", lw=1)
    ax.axvline(5.0, color="black", ls="--", lw=1)
    ax.annotate("top 5%\n(paper operating point)", xy=(5.0, 1.55),
                xytext=(6.5, 1.5), fontsize=8, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}%" for v in x])
    ax.set_xlabel("fraction of scored promoter windows retained (by D-Sites probability)")
    ax.set_ylabel("fold enrichment of DEP operons vs background")
    ax2.set_ylabel("-log10(Fisher p, one-sided greater)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title("FNR transfer — DEP-operon enrichment vs D-Sites score threshold\n"
                 "(operon level, identical promoter search space)")
    fig.tight_layout()
    fig.savefig(ec.FIGURES / "fnr_enrichment_vs_threshold.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
