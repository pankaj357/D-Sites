#!/usr/bin/env python3
"""
shared library for the D-Sites held-out evaluation pipeline
(held-out evaluation protocol

All heavy per-TF artefacts (window universe, feature matrix, site table,
window->site overlaps, promoter FASTA) are cached in results/cache/ so that
Parts A-E operate on byte-identical inputs.
"""

import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent                      # Revision/
ROOT = BASE.parent                            # D-Sites_/
DATA = BASE / "data"
RESULTS = BASE / "results"
CACHE = RESULTS / "cache"
FIGURES = BASE / "figures"
LOGS = RESULTS / "logs"
ORIG = ROOT / "Original" / "examples"
for d in (RESULTS, CACHE, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

from dsites import predictor as dp           # installed dsites v1.2.0

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SEED = 42
PAD = 10
UP = 300
DOWN = 50
NEG_RATIO = 5
N_SPLITS = 5
N_TREES = 300
PSEUDOCOUNT = 0.5
FIMO = "/Users/admin/meme/bin/fimo"
FIMO_THRESH = 1e-2
FIMO_DEFAULT_P = 1e-4        # FIMO's default significance threshold

FEATURE_NAMES = ["PWM_logodds", "GC", "Entropy",
                 "MGW_mean", "MGW_std", "PropTw_mean", "PropTw_std",
                 "Roll_mean", "Roll_std", "HelTw_mean", "HelTw_std"]

# Feature subsets for the ablation (Part D)
ABLATION_SETS = {
    "PWM_only": [0],
    "PWM_GC_Entropy": [0, 1, 2],
    "all_11": list(range(11)),
}

TF_CONFIG = {
    "AmrZ": dict(
        fasta=ORIG / "AmrZ" / "GCF_000006765.1_ASM676v1_genomic.fna",
        gff=ORIG / "AmrZ" / "sequence.gff3",
        motif=ORIG / "AmrZ" / "AmrZ.jaspar",
        tf_name="AmrZ", genome_acc="NC_002516.2", sites_source="collectf"),
    "GlxR": dict(
        fasta=ORIG / "GlxR" / "GCF_000010225.1_ASM1022v1_genomic.fna",
        gff=ORIG / "GlxR" / "genomic.gff",
        motif=ORIG / "GlxR" / "prodoric_mx000263_meme.txt",
        tf_name="GlxR", genome_acc="NC_009342.1", sites_source="collectf"),
    "CodY": dict(
        fasta=ORIG / "CodY" / "GCF_000008165.1_ASM816v1_genomic.fna",
        gff=ORIG / "CodY" / "genomic.gff",
        motif=ORIG / "CodY" / "CodY.meme.txt",
        tf_name="CodY", genome_acc="NC_005945.1", sites_source="collectf"),
    "FNR_ecoli": dict(
        fasta=DATA / "ecoli_K12_MG1655_genomic.fna",
        gff=DATA / "ecoli_K12_MG1655_genomic.gff",
        motif=ORIG / "Fnr" / "FNR_pfm.txt",
        tf_name="FNR", genome_acc="NC_000913.3", sites_source="regulondb"),
    "FNR_lt2": dict(
        fasta=ORIG / "Fnr" / "GCF_000006945.2_ASM694v2_genomic.fna",
        gff=ORIG / "Fnr" / "genomic.gff",
        motif=ORIG / "Fnr" / "FNR_pfm.txt",
        tf_name="FNR", genome_acc="NC_003197.2", sites_source=None),
}

# --------------------------------------------------------------------------
# Base/dinucleotide lookup tables (ACGT order, matching dsites predictor)
# --------------------------------------------------------------------------
_B2I = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    _B2I[_b] = _i

# NOTE (v1.3.0): shape features come from the published DNAshape
# pentamer query table via dsites.predictor.shape_stats_from_codes 

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_genome(fasta_path):
    from Bio import SeqIO
    contigs, contig_seqs = [], {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        contigs.append(rec.id)
        contig_seqs[rec.id] = str(rec.seq).upper()
    return contigs, contig_seqs


def load_motif_pwm(motif_path):
    """Return (pwm, motif_len) exactly as dsites v1.2.0 builds them."""
    motif = dp.read_motif(str(motif_path))
    pwm = dp.pfm_to_pwm(motif, pseudocount=PSEUDOCOUNT)
    return pwm, motif.length, motif


def load_sites(cfg):
    """Known-site table for a TF config. Returns DataFrame with columns
    site_id, contig, site_start, site_end, site_strand (+ source columns)."""
    if cfg["sites_source"] == "collectf":
        df = pd.read_csv(DATA / "collectf_clean.tsv", sep="\t")
        df = df[(df["TF"] == cfg["tf_name"]) &
                (df["genome_accession"] == cfg["genome_acc"])].copy()
        df = df.reset_index(drop=True)
        out = pd.DataFrame({
            "contig": df["genome_accession"],
            "site_start": df["site_start"].astype(int),
            "site_end": df["site_end"].astype(int),
            "site_strand": df["site_strand"],
        })
    elif cfg["sites_source"] == "regulondb":
        df = pd.read_csv(DATA / "regulondb_ecoli_FNR_sites.tsv", sep="\t")
        out = pd.DataFrame({
            "contig": cfg["genome_acc"],
            "site_start": df["left"].astype(int),
            "site_end": df["right"].astype(int),
            "site_strand": df["strand"],
        })
    else:
        raise ValueError("config has no known sites")
    out.insert(0, "site_id", np.arange(len(out)))
    return out


#--------------------------------------------------------------------------
#Positive extraction 
#--------------------------------------------------------------------------

def extract_positives(sites, contig_seqs, pwm, bg, k, pad=PAD):
    """Best-PWM k-mer within site coords +/- pad, both strands. Returns the
    sites DataFrame with kmer columns appended; sites with no valid k-mer
    are dropped (logged)."""
    rows = []
    n_drop = 0
    for r in sites.itertuples():
        seq = contig_seqs.get(r.contig)
        if seq is None:
            n_drop += 1
            continue
        s, e = int(r.site_start), int(r.site_end)
        a = max(1, s - pad)
        b = min(len(seq), e + pad)
        region = seq[a - 1:b]
        best = None
        best_kmer = best_pos = best_strand = None
        for i in range(0, len(region) - k + 1):
            for strand_kmer, strand in dp.get_both_strands(region[i:i + k]):
                sc = dp.logodds_score(strand_kmer, pwm, bg)
                if sc is None:
                    continue
                if best is None or sc > best:
                    best, best_kmer, best_pos, best_strand = sc, strand_kmer, a + i, strand
        if best_kmer is None:
            n_drop += 1
            continue
        rows.append(dict(r._asdict(), kmer=best_kmer, kmer_start=best_pos,
                         kmer_end=best_pos + k - 1, kmer_strand=best_strand,
                         kmer_score=best))
    if n_drop:
        print(f"[eval_common] WARNING: dropped {n_drop} sites without valid k-mer",
              flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Promoter window universe
# --------------------------------------------------------------------------
def promoter_regions(gff_path, contig_seqs, up=UP, down=DOWN):
    """Unique promoter regions (contig, start, end) covering -up/+down per CDS."""
    prom = dp.load_promoters(str(gff_path), up=up, down=down)
    regs = prom[prom["contig"].isin(contig_seqs.keys())][
        ["contig", "prom_start", "prom_end"]].drop_duplicates()
    regs = regs.sort_values(["contig", "prom_start", "prom_end"])
    return regs.reset_index(drop=True)


def build_window_universe(regions, contigs, contig_seqs, k):
    """Unique valid (all-ACGT) k-mer intervals across all promoter regions.
    Returns (win_contig_idx int16[W], win_pos int32[W] 1-based left coord)."""
    cidx_map = {c: i for i, c in enumerate(contigs)}
    ci_all, pos_all = [], []
    for r in regions.itertuples():
        seq = contig_seqs[r.contig]
        region_seq = seq[r.prom_start - 1:r.prom_end]
        occ = np.zeros(len(region_seq), dtype=bool)
        starts = dp.valid_window_starts(region_seq, occ, k)
        if len(starts) == 0:
            continue
        ci_all.append(np.full(len(starts), cidx_map[r.contig], dtype=np.int32))
        pos_all.append(starts.astype(np.int64) + r.prom_start)   # 1-based
    ci = np.concatenate(ci_all)
    pos = np.concatenate(pos_all)
    key = ci.astype(np.int64) << 40 | pos
    key_u = np.unique(key)          # np.unique returns sorted keys
    ci_u = (key_u >> 40).astype(np.int16)
    pos_u = (key_u & ((1 << 40) - 1)).astype(np.int32)
    return ci_u, pos_u


# --------------------------------------------------------------------------
# Vectorized feature computation 
# --------------------------------------------------------------------------
def codes_from_seqs(seqs):
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return _B2I[arr].reshape(len(seqs), -1)


def window_codes(win_ci, win_pos, contigs, contig_seqs, k):
    """(W,k) int8 base codes for the window universe, via stride tricks."""
    out = np.empty((len(win_pos), k), dtype=np.int8)
    for ci in np.unique(win_ci):
        c = contigs[ci]
        arr = np.frombuffer(contig_seqs[c].encode("ascii"), dtype=np.uint8)
        codes = _B2I[arr]
        swv = np.lib.stride_tricks.sliding_window_view(codes, k)
        m = win_ci == ci
        out[m] = swv[win_pos[m] - 1]
    return out


def features_from_codes(codes, pwm, bg):
    """codes: (n,k) int8 in 0..3 -> (n,11) float64 feature matrix with the
    predictor's exact feature semantics (v1.3.0: DNAshape PQT shape
    features via dp.shape_stats_from_codes)."""
    n, k = codes.shape
    lut = np.array([[math.log(pwm[i].get(b, 1e-9) / bg.get(b, 1e-9))
                     for b in "ACGT"] for i in range(k)], dtype=np.float64)
    llr = lut[np.arange(k)[None, :], codes].sum(axis=1)
    gc = ((codes == 1) | (codes == 2)).mean(axis=1)
    ent = np.zeros(n)
    for b in range(4):
        p = (codes == b).mean(axis=1)
        ent -= np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    X = np.empty((n, 11), dtype=np.float64)
    X[:, 0] = llr
    X[:, 1] = gc
    X[:, 2] = ent
    X[:, 3:] = dp.shape_stats_from_codes(codes)
    return X


def build_feature_matrix(wcodes, pwm, bg):
    """Interleaved both-strand features: row 2i = plus kmer, row 2i+1 =
    reverse-complement kmer (exactly the two examples the predictor emits)."""
    rc = (3 - wcodes[:, ::-1]).astype(np.int8)
    Xp = features_from_codes(wcodes, pwm, bg)
    Xm = features_from_codes(rc, pwm, bg)
    X = np.empty((2 * len(wcodes), 11), dtype=np.float32)
    X[0::2] = Xp
    X[1::2] = Xm
    return X


def kmer_features(kmers, pwm, bg):
    """Feature rows for explicit k-mer strings (training positives)."""
    return features_from_codes(codes_from_seqs([s.upper() for s in kmers]), pwm, bg)


# --------------------------------------------------------------------------
# Window <-> site overlaps
# --------------------------------------------------------------------------
def map_windows_to_sites(win_ci, win_pos, contigs, sites, k, pad=PAD):
    """Returns (win_ov, site2wins):
    win_ov: list; win_ov[i] = tuple of site_ids overlapped by window i
    site2wins: dict site_id -> int32 array of window indices."""
    win_ov = [()] * len(win_pos)
    site2wins = {}
    for ci, contig in enumerate(contigs):
        sub = sites[sites["contig"] == contig]
        if sub.empty:
            continue
        m = win_ci == ci
        widx_all = np.nonzero(m)[0]
        pos = win_pos[m].astype(np.int64)
        s_start = sub["site_start"].to_numpy(np.int64) - pad
        s_end = sub["site_end"].to_numpy(np.int64) + pad
        s_ids = sub["site_id"].to_numpy()
        chunk = 100_000
        for lo in range(0, len(pos), chunk):
            p = pos[lo:lo + chunk]
            ov = (p[:, None] <= s_end[None, :]) & (p[:, None] + k - 1 >= s_start[None, :])
            if not ov.any():
                continue
            wr, sr = np.nonzero(ov)
            for w, s in zip(wr, sr):
                gi = widx_all[lo + w]
                sid = int(s_ids[s])
                win_ov[gi] = win_ov[gi] + (sid,)
                site2wins.setdefault(sid, []).append(gi)
    site2wins = {s: np.array(v, dtype=np.int32) for s, v in site2wins.items()}
    return win_ov, site2wins


def labels_from_overlaps(win_ov):
    return np.array([1 if len(t) else 0 for t in win_ov], dtype=np.int8)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_path(tf):
    return CACHE / f"{tf}_windows.pkl"


def build_bundle(tf, force=False):
    """Build (or load) the cached per-TF bundle."""
    cp = cache_path(tf)
    if cp.exists() and not force:
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    cfg = TF_CONFIG[tf]
    contigs, contig_seqs = load_genome(cfg["fasta"])
    pwm, k, _motif = load_motif_pwm(cfg["motif"])
    bg = dp.genome_bg_freq(str(cfg["fasta"]))
    regions = promoter_regions(cfg["gff"], contig_seqs)
    win_ci, win_pos = build_window_universe(regions, contigs, contig_seqs, k)
    wcodes = window_codes(win_ci, win_pos, contigs, contig_seqs, k)
    X = build_feature_matrix(wcodes, pwm, bg)
    bundle = dict(
        tf=tf, cfg={kk: str(vv) for kk, vv in cfg.items()},
        contigs=contigs, k=k, bg=bg,
        regions=regions, win_ci=win_ci, win_pos=win_pos, X=X,
        build_seconds=time.time() - t0,
    )
    if cfg["sites_source"]:
        sites = load_sites(cfg)
        sites = extract_positives(sites, contig_seqs, pwm, bg, k)
        win_ov, site2wins = map_windows_to_sites(win_ci, win_pos, contigs, sites, k)
        bundle.update(sites=sites, win_ov=win_ov, site2wins=site2wins,
                      labels=labels_from_overlaps(win_ov))
    # keep genome sequences only in-memory for downstream FASTA writing;
    # cache them too (small genomes, ~5 MB) so all parts share one source.
    bundle["contig_seqs"] = contig_seqs
    with open(cp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    bundle["build_seconds"] = time.time() - t0
    return bundle


# --------------------------------------------------------------------------
# FIMO helpers
# --------------------------------------------------------------------------
def write_meme_motif(motif_path, out_path, name):
    """Rewrite any supported motif (JASPAR/MEME) as a minimal MEME Suite 4
    letter-probability matrix (row sums = 1)."""
    motif = dp.read_motif(str(motif_path))
    k = motif.length
    M = np.array([[float(motif.counts[b][i]) for b in "ACGT"] for i in range(k)],
                 dtype=np.float64)
    colsum = M.sum(axis=1, keepdims=True)
    colsum[colsum <= 0] = 1.0
    P = M / colsum
    nsites = max(1, int(round(float(M.sum(axis=1).max()))))
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", "",
             f"MOTIF {name} {name}", "",
             f"letter-probability matrix: alength= 4 w= {k} nsites= {nsites}"]
    for i in range(k):
        lines.append(" " + " ".join(f"{P[i, j]:.6f}" for j in range(4)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_bg_file(bg, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{b} {bg[b]:.6f}" for b in "ACGT") + "\n")
    return out_path


def write_promoter_fasta(regions, contig_seqs, out_path):
    """Unique promoter regions as FASTA; header '>{contig}:{start}-{end}'
    (1-based, inclusive) so FIMO coordinates map straight back to genomics."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in regions.itertuples():
            fh.write(f">{r.contig}:{r.prom_start}-{r.prom_end}\n")
            fh.write(contig_seqs[r.contig][r.prom_start - 1:r.prom_end] + "\n")
    return out_path


def fimo_version():
    return subprocess.run([FIMO, "--version"], capture_output=True, text=True
                          ).stdout.strip()


def run_fimo(meme_motif, promoters_fasta, bg_file, out_tsv, thresh=FIMO_THRESH):
    """Run FIMO on the promoter FASTA ONLY. Returns the exact command list."""
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    oc_dir = out_tsv.parent / (out_tsv.stem + "_oc")
    cmd = [FIMO, "--thresh", str(thresh), "--text", "--oc", str(oc_dir),
           "--bfile", str(bg_file), str(meme_motif), str(promoters_fasta)]
    with open(out_tsv, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True)
    return cmd


def parse_fimo_tsv(tsv_path):
    """Parse FIMO --text output -> DataFrame with genomic match coordinates
    gstart/gend (1-based inclusive, plus-strand coordinates).

    NOTE: our promoter FASTA headers have the form '>contig:start-end'.
    MEME Suite interprets that as a named sequence with a coordinate range
    and reports match start/stop as GENOMIC coordinates (verified
    empirically for FIMO 5.5.4: matched_sequence equals the genome slice at
    [start-1:stop], and sequence_name is reported as the bare contig id).
    run_fimo_fair validates that >=99% of matches fall inside the promoter
    regions, which would be impossible if coordinates were offset twice."""
    rows = []
    with open(tsv_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0].lower().startswith("motif"):      # header line
                continue
            if len(f) >= 10:                          # MEME 5.x layout
                seq_name, start, stop, strand, pval = f[2], f[3], f[4], f[5], f[7]
            elif len(f) >= 8:                         # classic 9-col layout
                seq_name, start, stop, strand, pval = f[1], f[2], f[3], f[4], f[6]
            else:
                continue
            start, stop, pval = int(start), int(stop), float(pval)
            lo, hi = min(start, stop), max(start, stop)
            rows.append((seq_name, lo, hi, strand, pval))
    return pd.DataFrame(rows, columns=["contig", "gstart", "gend", "strand", "pvalue"])


def fimo_scores_to_windows(fimo_df, win_ci, win_pos, contigs):
    """Map FIMO matches onto the window universe; per window keep the best
    (smallest) p-value. Returns float32 score array = -log10(p) (0 = no match
    at p < FIMO_THRESH)."""
    cidx_map = {c: i for i, c in enumerate(contigs)}
    key = (win_ci.astype(np.int64) << 40) | win_pos.astype(np.int64)
    # memory-light: argsort-based lookup
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    score = np.zeros(len(win_pos), dtype=np.float32)
    if len(fimo_df):
        q = (fimo_df["contig"].map(cidx_map).to_numpy(np.int64) << 40) | \
            fimo_df["gstart"].to_numpy(np.int64)
        p = -np.log10(np.clip(fimo_df["pvalue"].to_numpy(np.float64), 1e-300, 1.0))
        ordq = np.argsort(q, kind="stable")
        q, p = q[ordq], p[ordq]
        # per window keep max score
        uq, ui = np.unique(q, return_index=True)
        best = np.maximum.reduceat(p, ui)
        loc = np.searchsorted(key_sorted, uq)
        ok = (loc < len(key_sorted)) & (key_sorted[np.clip(loc, 0, len(key_sorted) - 1)] == uq)
        score[order[loc[ok]]] = best[ok].astype(np.float32)
    return score


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def f1_cutoff(y, scores):
    """F1-maximizing cutoff on (y, scores); used ONLY on RF out-of-bag
    probabilities (never on test-fold data)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    return (float(thr[i]) if len(thr) else 0.5), float(f1[:-1][i] if len(thr) else 0.0)


def window_site_metrics(prob, labels, scored_mask, test_site_ids, site2wins, cutoff):
    """Fold evaluation, exact definitions (see task spec / Methods):
    - windows overlapping a train-fold site are excluded (scored_mask).
    - TP window: prob>=cutoff and overlaps a test site; FP window: prob>=cutoff
      and overlaps no known site.
    - site RECOVERED: any scored window overlapping its padded coords has
      prob>=cutoff. recall = recovered / n_test_sites.
    - FPR = FP / (#scored windows overlapping no site).
    - PR-AUC = average_precision_score over all scored windows."""
    from sklearn.metrics import average_precision_score
    idx = np.nonzero(scored_mask)[0]
    p = prob[idx]
    l = labels[idx]
    pred = p >= cutoff
    tp = int(np.sum(pred & (l == 1)))
    fp = int(np.sum(pred & (l == 0)))
    n_neg = int(np.sum(l == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    recovered = 0
    for sid in test_site_ids:
        w = site2wins.get(int(sid))
        if w is None:
            continue
        wsc = w[scored_mask[w]]
        if len(wsc) and np.any(prob[wsc] >= cutoff):
            recovered += 1
    n_test = len(test_site_ids)

    in_space = 0
    scorable = 0
    for sid in test_site_ids:
        w = site2wins.get(int(sid))
        if w is None:
            continue
        in_space += 1
        if np.any(scored_mask[w]):
            scorable += 1
    recall = recovered / n_test if n_test else 0.0
    recall_in_space = recovered / in_space if in_space else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ap = float(average_precision_score(l, p)) if l.sum() else 0.0
    # top 5% by probability 
    n_top = max(1, int(0.05 * len(idx)))
    order = np.argsort(-p, kind="stable")[:n_top]
    top_idx = idx[order]
    tp5 = int(np.sum(labels[top_idx] == 1))
    prec5 = tp5 / n_top
    rec5_sites = 0
    top_mask = np.zeros(len(prob), dtype=bool)
    top_mask[top_idx] = True
    for sid in test_site_ids:
        w = site2wins.get(int(sid))
        if w is None:
            continue
        wsc = w[scored_mask[w]]
        if len(wsc) and np.any(top_mask[wsc]):
            rec5_sites += 1
    rec5 = rec5_sites / n_test if n_test else 0.0
    rec5_in_space = rec5_sites / in_space if in_space else 0.0
    return dict(cutoff=float(cutoff), n_scored=int(len(idx)),
                n_pos_windows=int(l.sum()), n_neg_windows=n_neg,
                TP=tp, FP=fp, precision=float(precision), recall=float(recall),
                F1=float(f1), FPR=float(fpr), PR_AUC=ap,
                n_test_sites=int(n_test), recovered_sites=int(recovered),
                n_test_sites_in_space=int(in_space),
                n_test_sites_scorable=int(scorable),
                recall_in_space=float(recall_in_space),
                top5pct_n=int(n_top), top5pct_TP=tp5,
                top5pct_precision=float(prec5), top5pct_recall=float(rec5),
                top5pct_recall_in_space=float(rec5_in_space),
                top5pct_recovered_sites=int(rec5_sites))


def mean_sd(per_fold, key):
    v = np.array([f[key] for f in per_fold], dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# --------------------------------------------------------------------------
# Runinfo / IO
# --------------------------------------------------------------------------
def runinfo(extra=None):
    import platform
    import sklearn
    import Bio
    info = dict(
        python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, biopython=Bio.__version__,
        dsites=dp.__version__, fimo=None, seed=SEED, pad=PAD, up=UP, down=DOWN,
        neg_ratio=NEG_RATIO, pseudocount=PSEUDOCOUNT,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    try:
        info["fimo"] = fimo_version()
    except Exception:
        pass
    if extra:
        info.update(extra)
    return info


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


# --------------------------------------------------------------------------
# Self-test: vectorized features must match the predictor's functions
# --------------------------------------------------------------------------
def selftest(n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    cfg = TF_CONFIG["AmrZ"]
    pwm, k, _ = load_motif_pwm(cfg["motif"])
    bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    seqs = ["".join(rng.choice(list("ACGT"), size=k)) for _ in range(n)]
    codes = codes_from_seqs(seqs)
    Xv = features_from_codes(codes, pwm, bg)
    Xref = []
    for s in seqs:
        Xref.append([dp.logodds_score(s, pwm, bg),
                     (s.count("G") + s.count("C")) / len(s),
                     dp.seq_entropy(s)] + dp.dinuc_shape_stats(s))
    Xref = np.array(Xref)
    assert np.allclose(Xv, Xref, rtol=0, atol=1e-10), \
        f"feature mismatch: max abs diff {np.abs(Xv - Xref).max()}"
    # both-strand interleaving: revcomp rows must match predictor on revcomp
    from Bio.Seq import Seq as BSeq
    X2 = build_feature_matrix(codes, pwm, bg)
    for j in (0, n - 1):
        rc = str(BSeq(seqs[j]).reverse_complement())
        ref = [dp.logodds_score(rc, pwm, bg),
               (rc.count("G") + rc.count("C")) / len(rc),
               dp.seq_entropy(rc)] + dp.dinuc_shape_stats(rc)
        assert np.allclose(X2[2 * j + 1], np.array(ref), rtol=0, atol=1e-5)
        ref_p = [dp.logodds_score(seqs[j], pwm, bg),
                 (seqs[j].count("G") + seqs[j].count("C")) / len(seqs[j]),
                 dp.seq_entropy(seqs[j])] + dp.dinuc_shape_stats(seqs[j])
        assert np.allclose(X2[2 * j], np.array(ref_p), rtol=0, atol=1e-5)
    print(f"[selftest] PASSED: vectorized features identical to dsites v{dp.__version__} "
          f"predictor functions on {n} random {k}-mers (both strands).", flush=True)


if __name__ == "__main__":
    selftest()
