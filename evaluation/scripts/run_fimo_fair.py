#!/usr/bin/env python3
"""
FIMO baseline.

Outputs (results/):
  fimo_{TF}.tsv           raw FIMO --text output
  fimo_{TF}.meme, fimo_{TF}.bg, promoters_{TF}.fasta   inputs as run
  fimo_scores_{TF}.npz    per-window -log10(p) scores + labels (for Part C)
  fimo_fair_{TF}.json     metrics, matched-count comparison, command, version

Usage:
  python run_fimo_fair.py --tf AmrZ [--tf GlxR ...]
  python run_fimo_fair.py --all
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import eval_common as ec

FIMO_SCORE_CUTOFF = -np.log10(ec.FIMO_DEFAULT_P)   # 4.0  <=>  p = 1e-4


def run_fimo_fair(tf, outdir=ec.RESULTS, verbose=True):
    t0 = time.time()
    outdir = Path(outdir)
    bundle = ec.build_bundle(tf)
    if "sites" not in bundle:
        raise SystemExit(f"{tf}: no known sites configured")
    sites = bundle["sites"]
    labels = bundle["labels"]
    site2wins = bundle["site2wins"]
    contigs = bundle["contigs"]

    # ---------------- FIMO inputs ----------------
    meme = ec.write_meme_motif(bundle["cfg"]["motif"], outdir / f"fimo_{tf}.meme", tf)
    bgf = ec.write_bg_file(bundle["bg"], outdir / f"fimo_{tf}.bg")
    pfasta = outdir / f"promoters_{tf}.fasta"
    if not pfasta.exists():
        ec.write_promoter_fasta(bundle["regions"], bundle["contig_seqs"], pfasta)

    # ---------------- run FIMO ----------------
    tsv = outdir / f"fimo_{tf}.tsv"
    cmd = ec.run_fimo(meme, pfasta, bgf, tsv)
    ver = ec.fimo_version()
    if verbose:
        print(f"[{tf}] FIMO {ver} done: {' '.join(cmd)}", flush=True)

    # ---------------- map to window universe ----------------
    fdf = ec.parse_fimo_tsv(tsv)
    # validate the genomic-coordinate interpretation of FIMO output
    _iv = bundle["regions"][["prom_start", "prom_end"]].to_numpy()
    _iv = _iv[np.argsort(_iv[:, 1])]
    _ends = _iv[:, 1]
    _pos = np.searchsorted(_ends, fdf["gstart"].to_numpy())
    _ok = _pos < len(_iv)
    _inside = np.zeros(len(fdf), dtype=bool)
    _inside[_ok] = fdf["gstart"].to_numpy()[_ok] >= _iv[_pos[_ok], 0]
    frac_inside = float(_inside.mean()) if len(fdf) else 1.0
    assert frac_inside > 0.99, (
        f"FIMO coordinate sanity check failed for {tf}: only "
        f"{frac_inside:.3f} of matches fall inside promoter regions")
    fscore = ec.fimo_scores_to_windows(fdf, bundle["win_ci"], bundle["win_pos"], contigs)
    np.savez_compressed(outdir / f"fimo_scores_{tf}.npz",
                        score=fscore, label=labels,
                        win_ci=bundle["win_ci"], win_pos=bundle["win_pos"])

    # ---------------- global metrics at FIMO default threshold ----------------
    scored = np.ones(len(fscore), dtype=bool)          # one global run
    all_site_ids = sites["site_id"].to_numpy()
    m_def = ec.window_site_metrics(fscore, labels, scored, all_site_ids,
                                   site2wins, FIMO_SCORE_CUTOFF)
    ap = float(average_precision_score(labels, fscore))
    n_pred = int(np.sum(fscore >= FIMO_SCORE_CUTOFF))

    # ---------------- matched-count comparison ----------------
    N = n_pred
    oof_path = outdir / f"oof_{tf}.npz"
    matched = None
    reverse = None
    saturation = None
    if oof_path.exists():
        oof = np.load(oof_path)
        oof_score, oof_label = oof["score"], oof["label"]
        # map pooled-OOF windows back to universe indices for site recovery
        key_all = (bundle["win_ci"].astype(np.int64) << 40) | bundle["win_pos"].astype(np.int64)
        key_oof = (oof["win_ci"].astype(np.int64) << 40) | oof["win_pos"].astype(np.int64)
        order = np.argsort(key_all, kind="stable")
        pos = np.searchsorted(key_all[order], key_oof)
        oof_widx = order[pos]
        # D-Sites top-N by pooled OOF probability
        n_top = min(N, len(oof_score))
        top = np.argsort(-oof_score, kind="stable")[:n_top]
        ds_tp = int(np.sum(oof_label[top] == 1))
        ds_prec = ds_tp / n_top if n_top else 0.0
        ds_top_mask = np.zeros(len(fscore), dtype=bool)
        ds_top_mask[oof_widx[top]] = True
        # FIMO top-N by p-value (== all windows passing p<1e-4)
        f_top_mask = fscore >= FIMO_SCORE_CUTOFF
        f_tp = int(np.sum(f_top_mask & (labels == 1)))
        f_prec = f_tp / N if N else 0.0

        def site_recall(mask):
            rec = rec_is = 0
            n_is = 0
            for sid in all_site_ids:
                w = site2wins.get(int(sid))
                if w is None:
                    continue
                n_is += 1
                if np.any(mask[w]):
                    rec += 1
                    rec_is += 1
            return (rec / len(all_site_ids), rec_is / n_is if n_is else 0.0,
                    rec, n_is)

        ds_rec, ds_rec_is, ds_nrec, n_is = site_recall(ds_top_mask)
        f_rec, f_rec_is, f_nrec, _ = site_recall(f_top_mask)
        # RF probability-saturation check (are top ranks tie-degenerate?)
        saturation = {f"n_windows_pooled_prob_ge_{q}": int(np.sum(oof_score >= q))
                      for q in (0.999999, 0.99, 0.9)}
        matched = dict(
            N=int(N), note="FIMO is one global run; D-Sites uses pooled "
                           "out-of-fold probabilities from grouped CV.",
            dsites=dict(top_n=int(n_top), TP_windows=ds_tp,
                        precision=ds_prec, recall=ds_rec,
                        recall_in_space=ds_rec_is, recovered_sites=ds_nrec),
            fimo=dict(top_n=int(N), TP_windows=f_tp,
                      precision=f_prec, recall=f_rec,
                      recall_in_space=f_rec_is, recovered_sites=f_nrec),
            n_sites=int(len(all_site_ids)), n_sites_in_space=int(n_is))
        # REVERSE matched-count: N2 = size of the D-Sites top-5% operating
        # set; both methods contribute their top-N2 windows.
        N2 = max(1, int(0.05 * len(oof_score)))
        ds_top2 = np.argsort(-oof_score, kind="stable")[:N2]
        ds2_mask = np.zeros(len(fscore), dtype=bool)
        ds2_mask[oof_widx[ds_top2]] = True
        f_top2 = np.argsort(-fscore, kind="stable")[:N2]
        f2_mask = np.zeros(len(fscore), dtype=bool)
        f2_mask[f_top2] = True
        ds2_tp = int(np.sum(oof_label[ds_top2] == 1))
        f2_tp = int(np.sum(labels[f_top2] == 1))
        ds2_rec, ds2_rec_is, ds2_nrec, _ = site_recall(ds2_mask)
        f2_rec, f2_rec_is, f2_nrec, _ = site_recall(f2_mask)
        reverse = dict(
            N=int(N2), note="Reverse matched-count: N = size of the D-Sites "
                             "top-5% operating set; both methods contribute "
                             "their top-N windows on the identical search "
                             "space. FIMO scores below its reporting "
                             "threshold (p>=1e-2) are 0; if N exceeds the "
                             "reported matches, remaining FIMO ranks are "
                             "arbitrary (score-0 ties) but cannot gain "
                             "recall beyond the p<1e-2 set.",
            dsites=dict(top_n=int(N2), TP_windows=ds2_tp,
                        precision=ds2_tp / N2, recall=ds2_rec,
                        recall_in_space=ds2_rec_is,
                        recovered_sites=ds2_nrec),
            fimo=dict(top_n=int(N2), TP_windows=f2_tp,
                      precision=f2_tp / N2, recall=f2_rec,
                      recall_in_space=f2_rec_is,
                      recovered_sites=f2_nrec),
            n_sites=int(len(all_site_ids)), n_sites_in_space=int(n_is))
    else:
        print(f"[{tf}] WARNING: {oof_path} missing — run cross_validation.py "
              f"--tf {tf} first; matched-count comparison skipped.", flush=True)

    results = dict(
        tf=tf,
        note=("FIMO is ONE GLOBAL RUN on the identical promoter search space; "
              "it is compared against D-Sites POOLED grouped-CV results, not "
              "per-fold numbers. Unreported positions (p >= 1e-2) are treated "
              "as score 0 (--thresh 1e-2)."),
        n_windows=int(len(fscore)), n_sites=int(len(all_site_ids)),
        n_fimo_rows=int(len(fdf)),
        fimo_matches_inside_promoters_frac=frac_inside,
        fimo_default_threshold_p=ec.FIMO_DEFAULT_P,
        metrics_at_default=m_def, PR_AUC=ap,
        n_predictions_at_default=n_pred,
        matched_count=matched,
        matched_count_reverse_at_dsites_top5pct=reverse,
        dsites_prob_saturation=saturation,
        fimo_version=ver, command=cmd,
        runinfo=ec.runinfo(dict(wall_seconds=round(time.time() - t0, 1))))
    ec.save_json(results, outdir / f"fimo_fair_{tf}.json")
    if verbose:
        print(f"[{tf}] DONE in {time.time()-t0:.0f}s | FIMO@1e-4: "
              f"recall={m_def['recall']:.3f} (in-space {m_def['recall_in_space']:.3f}) "
              f"precision={m_def['precision']:.4f} PR-AUC={ap:.4f} N={n_pred}",
              flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", action="append",
                    choices=["AmrZ", "GlxR", "CodY", "FNR_ecoli"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    tfs = ["AmrZ", "GlxR", "CodY"] if args.all else (args.tf or [])
    if not tfs:
        ap.error("--tf or --all required")
    for tf in tfs:
        run_fimo_fair(tf)


if __name__ == "__main__":
    main()
