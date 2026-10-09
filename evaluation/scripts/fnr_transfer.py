#!/usr/bin/env python3
"""
fnr_transfer.py — Part E: FNR cross-genome transfer (E. coli K-12 ->
Salmonella Typhimurium LT2) for the D-Sites revision.

Steps:
1. TRAINING DATA (E. coli K-12 MG1655, NC_000913.3): positives = the 97
   documented RegulonDB FNR sites (Revision/data/regulondb_ecoli_FNR_sites.tsv),
   each represented by the best-PWM-scoring 14-mer within site coords +/-10 bp
   (both strands), exactly as the predictor extracts positives; negatives =
   5 x 97 windows sampled (seed 42) from E. coli promoter windows (-300/+50
   per CDS, both strands), excluding +/-10 bp around ALL 97 known sites.
2a. INTERNAL VALIDATION: the same grouped 5-fold CV as Part A within the 97
    E. coli sites -> results/cv_FNR_ecoli.json (written by
    cross_validation.run_cv; leakage assertion included).
2b. FAIR FIMO BASELINE: FIMO with the same FNR motif (PRODORIC MX000004) on
    the E. coli promoter FASTA only -> results/fimo_fair_FNR_ecoli.json
    (written by run_fimo_fair).
3. TRANSFER: a final RandomForestClassifier(n_estimators=300, n_jobs=-1,
   random_state=42, oob_score=True) is trained on ALL 97 E. coli sites
   (cutoff = F1-max on OOB probabilities) and applied to scan Salmonella LT2
   promoters (NC_003197.2 + NC_003277.2 from Original/examples/Fnr/):
     - results/fnr_transfer_predictions.csv        (prob >= OOB-F1 cutoff)
     - results/fnr_transfer_predictions_top5pct.csv(top 5% of scored windows)
     - results/fnr_lt2_window_scores.csv.gz        (every scored window)
   FIMO with the same motif is run on the same LT2 promoter FASTA:
     - results/fimo_fnr_lt2_promoters.tsv          (raw FIMO output)
   These feed the Phase-4 operon-level test; NO enrichment is computed here.
4. results/fnr_transfer_runinfo.json records all commands, versions, seeds,
   cutoffs and counts.

Usage:
  python fnr_transfer.py [--folds 5 --trees 300 --n_jobs -1]
  python fnr_transfer.py --smoke     # 2 folds / 50 trees for CV, 50 trees final
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq as BSeq
from sklearn.ensemble import RandomForestClassifier

import eval_common as ec
from cross_validation import run_cv
from run_fimo_fair import run_fimo_fair


def train_final_model(bundle, n_trees, seed, n_jobs):
    """Train on ALL E. coli FNR sites; return (clf, cutoff, oob_f1, counts)."""
    sites = bundle["sites"]
    labels = bundle["labels"]
    pwm, k, _ = ec.load_motif_pwm(bundle["cfg"]["motif"])
    X_pos = ec.kmer_features(sites["kmer"].tolist(), pwm, bundle["bg"])
    rng = np.random.default_rng(seed)
    eligible = np.nonzero(labels == 0)[0]
    n_neg = ec.NEG_RATIO * len(sites)
    sel = eligible[rng.choice(len(eligible), size=n_neg, replace=False)]
    neg_rows = np.sort(np.concatenate([2 * sel, 2 * sel + 1]))
    X_tr = np.vstack([X_pos, bundle["X"][neg_rows]])
    y_tr = np.concatenate([np.ones(len(sites), int),
                           np.zeros(len(neg_rows), int)])
    clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs,
                                 random_state=seed, oob_score=True)
    clf.fit(X_tr, y_tr)
    oob = clf.oob_decision_function_[:, 1]
    ok = ~np.isnan(oob)
    cutoff, oob_f1 = ec.f1_cutoff(y_tr[ok], oob[ok])
    return clf, cutoff, oob_f1, dict(n_pos=len(sites), n_neg_rows=len(neg_rows),
                                     oob_score=float(clf.oob_score_))


def scan_and_write(clf, cutoff, bundle_lt2, outdir, tag=""):
    """Scan LT2 promoter windows; write prediction CSVs."""
    outdir = Path(outdir)
    X = bundle_lt2["X"]
    k = bundle_lt2["k"]
    contigs = bundle_lt2["contigs"]
    contig_seqs = bundle_lt2["contig_seqs"]
    t0 = time.time()
    p2 = clf.predict_proba(X)[:, 1].astype(np.float32)
    prob = np.maximum(p2[0::2], p2[1::2])
    strand_is_minus = p2[1::2] > p2[0::2]
    win_ci, win_pos = bundle_lt2["win_ci"], bundle_lt2["win_pos"]

    all_df = pd.DataFrame({
        "contig": np.array(contigs, dtype=object)[win_ci],
        "start": win_pos,
        "end": win_pos + k - 1,
        "strand": np.where(strand_is_minus, "-", "+"),
        "score": prob,
    })
    all_df = all_df.sort_values("score", ascending=False, kind="stable")
    all_df.to_csv(outdir / f"fnr_lt2_window_scores{tag}.csv.gz", index=False,
                  compression="gzip")

    def add_seq(df):
        seqs = []
        for r in df.itertuples():
            s = contig_seqs[r.contig][r.start - 1:r.end]
            seqs.append(str(BSeq(s).reverse_complement()) if r.strand == "-" else s)
        return df.assign(seq=seqs)

    pred = all_df[all_df["score"] >= cutoff]
    add_seq(pred).to_csv(outdir / f"fnr_transfer_predictions{tag}.csv", index=False)
    n_top = max(1, int(0.05 * len(all_df)))
    top = all_df.head(n_top)
    add_seq(top).to_csv(outdir / f"fnr_transfer_predictions_top5pct{tag}.csv",
                        index=False)
    return dict(n_windows_scored=int(len(all_df)), cutoff=float(cutoff),
                n_predictions=int(len(pred)), n_top5pct=int(len(top)),
                scan_seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=ec.N_SPLITS)
    ap.add_argument("--trees", type=int, default=ec.N_TREES)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_splits = 2 if args.smoke else args.folds
    n_trees = 50 if args.smoke else args.trees
    outdir = ec.RESULTS / "smoke_test" if args.smoke else ec.RESULTS
    tag = "_smoke" if args.smoke else ""
    runinfo = dict(commands={}, seeds=dict(global_seed=ec.SEED))

    # ---------------- 1+2a: internal grouped CV on the 97 E. coli sites ----
    cv = run_cv("FNR_ecoli", n_splits=n_splits, n_trees=n_trees,
                n_jobs=args.n_jobs, outdir=outdir, tag=tag)
    runinfo["internal_cv_json"] = str(Path(outdir) / f"cv_FNR_ecoli{tag}.json")

    # ---------------- 2b: fair FIMO baseline on E. coli promoters ----------
    if not args.smoke:
        fm = run_fimo_fair("FNR_ecoli", outdir=outdir)
        runinfo["commands"]["fimo_ecoli"] = fm["command"]
    else:
        print("[smoke] FIMO E. coli baseline skipped in smoke mode", flush=True)

    # ---------------- 3: transfer to Salmonella LT2 ------------------------
    b_ec = ec.build_bundle("FNR_ecoli")
    b_lt2 = ec.build_bundle("FNR_lt2")
    clf, cutoff, oob_f1, counts = train_final_model(b_ec, n_trees, ec.SEED,
                                                    args.n_jobs)
    print(f"[FNR transfer] final model: OOB acc={counts['oob_score']:.3f}, "
          f"OOB-F1 cutoff={cutoff:.4f}", flush=True)
    scan = scan_and_write(clf, cutoff, b_lt2, outdir, tag=tag)

    # FIMO on the same LT2 promoter FASTA
    meme = ec.write_meme_motif(b_lt2["cfg"]["motif"],
                               Path(outdir) / f"fimo_FNR_lt2{tag}.meme", "FNR")
    bgf = ec.write_bg_file(b_lt2["bg"], Path(outdir) / f"fimo_FNR_lt2{tag}.bg")
    pfasta = Path(outdir) / f"promoters_FNR_lt2{tag}.fasta"
    if not pfasta.exists():
        ec.write_promoter_fasta(b_lt2["regions"], b_lt2["contig_seqs"], pfasta)
    lt2_tsv = Path(outdir) / (f"fimo_fnr_lt2_promoters{tag}.tsv" if tag
                              else "fimo_fnr_lt2_promoters.tsv")
    cmd = ec.run_fimo(meme, pfasta, bgf, lt2_tsv)
    runinfo["commands"]["fimo_lt2"] = cmd
    n_fimo_rows = len(ec.parse_fimo_tsv(lt2_tsv))
    print(f"[FNR transfer] FIMO on LT2 promoters: {n_fimo_rows} matches "
          f"(p<{ec.FIMO_THRESH})", flush=True)

    # ---------------- 4: runinfo -------------------------------------------
    runinfo.update(dict(
        note="Transfer scan + FIMO outputs feed the Phase-4 operon-level "
             "test; no enrichment is computed here.",
        final_model=dict(n_trees=n_trees, n_jobs=str(args.n_jobs),
                         cutoff_oob_f1=cutoff, oob_f1=oob_f1, **counts),
        transfer=scan, fimo_lt2_rows=n_fimo_rows,
        outputs=[str(p) for p in [
            Path(outdir) / f"cv_FNR_ecoli{tag}.json",
            Path(outdir) / f"fnr_transfer_predictions{tag}.csv",
            Path(outdir) / f"fnr_transfer_predictions_top5pct{tag}.csv",
            Path(outdir) / f"fnr_lt2_window_scores{tag}.csv.gz",
            lt2_tsv]],
        wall_seconds=round(time.time() - t0, 1)))
    runinfo.update(ec.runinfo())
    ec.save_json(runinfo, Path(outdir) / f"fnr_transfer_runinfo{tag}.json")
    print(f"[FNR transfer] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
