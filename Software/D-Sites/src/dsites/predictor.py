#!/usr/bin/env python3
"""
D-Sites: Hybrid TFBS predictor combining PWM, dinucleotide shape, and Random Forest

Author:
Pankaj, Indian Agricultural Research Institute (IARI), ft.pank@gmail.com  
Kanaka KK, Scientist, Animal Genetics & Breeding, ICAR-IIAB, India ,kkokay07@gmail.com 
License: MIT

Version: 1.3.0
Date: 2025

Description:
-------------
This tool predicts transcription factor binding sites (TFBS) in bacterial genomes
using a hybrid computational framework that combines:

1. Position Weight Matrix (PWM) scoring with count-based pseudocount adjustment
2. Dinucleotide DNA-shape features (MGW, Propeller Twist, Roll, Helical Twist)
3. Sequence-derived features (GC content, sequence entropy)
4. Random Forest classifier for supervised prediction

The pipeline scans promoter windows (or, optionally, the full genome) on ALL
contigs present in the input FASTA in a memory-efficient, batch-parallel manner
and allows automatic probability cutoff selection via precision-recall
optimization.

Methodological notes (v1.2.0):
------------------------------
- PWM probabilities use a standard count-based pseudocount:
  p_b = (count_b + pc) / (N + 4*pc), where N is the number of aligned sites
  contributing to the column and pc is the pseudocount (default 0.5).
  All log-odds scores use the NATURAL logarithm.
- Dinucleotide shape parameters are REPLACED (v1.3.0) by the published
  DNAshape pentamer query table (Zhou et al. 2013, NAR 41:W56-W62, as
  distributed in DNAshapeR v1.38.0), reproducing DNAshapeR::getShape
  per-position values exactly; the previous hardcoded 16-dinucleotide
  table (untraceable provenance, ad-hoc strand symmetrization) is removed.
- Negative training windows are sampled from the SAME windows that are scanned
  (promoter windows by default; entire contigs with --full_genome), excluding
  +/-pad bp around known positive sites, and the requested --neg_ratio is
  honored exactly (no hidden minimum unless --min_negatives is set).
- All randomness is controlled by --seed (python `random` and NumPy RNG), so
  runs are fully deterministic.
- Window feature extraction / prediction is parallelized across batches with
  joblib.Parallel(n_jobs=-1).

Publication Notes:
-------------------
- TFBS prediction combines sequence-specific and DNA structural features for improved accuracy.
- Optimized negative sampling reduces bias and improves classifier training.
- Batch-based genome scanning enables large genome analysis without memory overhead.
- The tool provides output as CSV with predicted binding sites, strand orientation, and probability score.

References:
-----------
- Olson WK, Gorin AA, Lu XJ, Hock LM, Zhurkin VB (1998) DNA sequence-dependent
  deformability deduced from protein-DNA crystal complexes.
  Proc Natl Acad Sci USA 95:11163-11168. (dinucleotide shape parameters)
- Zhou T, Yang L, Lu Y, Dror I, Dantas Machado AC, Ghane T, Di Felice R,
  Rohs R (2013) DNAshape: a method for the high-throughput prediction of DNA
  structural features on a genomic scale. Nucleic Acids Res 41:W56-W62.
- Random Forest supervised classifier for TFBS prediction.
- PWM scoring methodology with pseudocount adjustment.

"""

import os
import sys
import math
import random
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq as BSeq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
import importlib.resources



__version__ = "1.3.0"

# ---------------------------
# Bundled known-site dataset (CollecTF export shipped inside the package)
# ---------------------------
KNOWN_DATASET_RESOURCE = importlib.resources.files("dsites").joinpath("collectf_export.tsv")
if not KNOWN_DATASET_RESOURCE.is_file():
    raise FileNotFoundError(
        "Could not locate collectf_export.tsv inside the installed dsites package. "
        "The package installation appears broken; reinstall dsites or provide known "
        "sites explicitly with --known."
    )
KNOWN_DATASET_PATH = str(KNOWN_DATASET_RESOURCE)

# ---------------------------
# DNA SHAPE FEATURES: published pentamer query table (PQT)
# ---------------------------
# v1.3.0: the previous hardcoded 16-dinucleotide DINUC_PARAMS table (values
# of untraceable provenance, strand-symmetrized ad hoc) has been REPLACED by
# the published DNAshape pentamer query table (Zhou et al. 2013, Nucleic
# Acids Res 41:W56-W62) as distributed with DNAshapeR v1.38.0 (Bioconductor;
# Chiu et al. 2016, Bioinformatics 32:1211-1213), decoded from DNAshapeR's
# C++ sources (TableCompiler.cpp QueryTable array; column identities from
# properties.cpp; lookup/step conventions from prediction.cpp and map.cpp).
# The swap is justified by a protocol-identical A/B test (held-out grouped
# CV, AmrZ/CodY): equal-to-better performance vs the old table -- see
# Revision/results/shape_ab_test/ and Revision/results/shape_ab_test/
# pqt_python_vs_R_verification.json (this implementation reproduces
# DNAshapeR::getShape per-position values to <1e-6).
#
# Conventions replicated EXACTLY from DNAshapeR (see prediction.cpp):
#   - the table stores ONE strand per reverse-complement pair; a pentamer
#     found via its reverse complement (status -1) uses the same MGW/ProT
#     value but the two step values with roles swapped (Roll1<->Roll2,
#     HelT1<->HelT2); no sign flips for these four parameters.
#   - MGW/ProT are central-bp values of each pentamer -> valid at bp
#     2..L-3 (0-based); terminal 2 bp at each end get no value.
#   - Roll/HelT are bp-step parameters; pentamer starting at p contributes
#     its first step value to step p+1 and its second to step p+2; interior
#     steps (2..L-4) are the MEAN of the two contributing pentamers; the
#     first valid step (1) uses only pentamer 0's first value and the last
#     (L-3) only the last pentamer's second value.
#   - getShape prints values with 2 decimals; per-position values here are
#     rounded likewise (np.round half-to-even) before aggregation.
PQT_RESOURCE = importlib.resources.files("dsites").joinpath("dnashape_pqt.tsv")
if not PQT_RESOURCE.is_file():
    raise FileNotFoundError(
        "Could not locate dnashape_pqt.tsv inside the installed dsites package. "
        "The package installation appears broken; reinstall dsites."
    )
PQT_PATH = str(PQT_RESOURCE)


def _load_dnashape_pqt(path):
    """Parse dnashape_pqt.tsv -> dict pentamer -> (MGW, ProT, Roll1, Roll2,
    HelT1, HelT2)."""
    table = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] == "pentamer":
                continue
            table[f[0]] = tuple(float(x) for x in f[1:7])
    if len(table) != 512:
        raise ValueError(f"dnashape_pqt.tsv: expected 512 pentamers, got {len(table)}")
    return table


_PQT_DICT = _load_dnashape_pqt(PQT_PATH)

# Pentamer codes use the pipeline's base order A=0, C=1, G=2, T=3:
# code = c0*256 + c1*64 + c2*16 + c3*4 + c4  (0..1023)
_PQT_VALUES = np.zeros((512, 6), dtype=np.float64)
_PQT_ROW = np.full(1024, -1, dtype=np.int32)
_PQT_STATUS = np.zeros(1024, dtype=np.int8)   # +1 direct, -1 via reverse complement


def _pentamer_code(seq):
    code = 0
    for b in seq:
        code = code * 4 + "ACGT".index(b)
    return code


def _revcomp_code(code, length=5):
    rc = 0
    for _ in range(length):
        rc = rc * 4 + (3 - (code & 3))
        code >>= 2
    return rc


for _i, (_pent, _vals) in enumerate(sorted(_PQT_DICT.items())):
    _PQT_VALUES[_i] = _vals
    _PQT_ROW[_pentamer_code(_pent)] = _i
    _PQT_STATUS[_pentamer_code(_pent)] = 1
for _c in range(1024):
    if _PQT_ROW[_c] < 0:
        _rc = _revcomp_code(_c)
        if _PQT_ROW[_rc] >= 0:
            _PQT_ROW[_c] = _PQT_ROW[_rc]
            _PQT_STATUS[_c] = -1
assert (_PQT_ROW >= 0).all(), "PQT must cover every ACGT pentamer (via revcomp)"


def _round2(x):
    """Round to 2 decimals EXACTLY as C++ iostream/printf (setprecision(2),
    fixed) does: correctly-rounded decimal conversion of the binary double
    (round-half-even). np.round is NOT equivalent (it rounds x*100, which
    introduces scaling error on .xx5 boundaries -- this caused 0.01-level
    discrepancies vs DNAshapeR on ~5% of averaged step values). Uses
    Python's correctly-rounded float formatting; only applied to the
    averaged step values (table values are already 2-decimal)."""
    flat = np.ravel(x)
    out = np.array([float(f"{v:.2f}") for v in flat], dtype=np.float64)
    return out.reshape(x.shape)


def shape_arrays_from_codes(codes):
    """Per-position shape values replicating DNAshapeR::getShape exactly.

    codes: (k,) or (n, k) int array of ACGT codes (A=0,C=1,G=2,T=3).
    Returns (mgw, prot, roll, helt): for a k-mer, mgw/prot have k-4 values
    (central bp of each pentamer, valid bp 2..k-3), roll/helt have k-3
    values (valid steps 1..k-3). Values rounded to 2 decimals as getShape
    prints them."""
    single = codes.ndim == 1
    if single:
        codes = codes[None, :]
    codes = codes.astype(np.int32, copy=False)   # int8 inputs overflow *256
    n, k = codes.shape
    pc = (codes[:, :-4] * 256 + codes[:, 1:-3] * 64 + codes[:, 2:-2] * 16
          + codes[:, 3:-1] * 4 + codes[:, 4:])
    v = _PQT_VALUES[_PQT_ROW[pc]]          # (n, k-4, 6)
    st = _PQT_STATUS[pc]                   # (n, k-4)
    mgw = v[..., 0]
    prot = v[..., 1]
    va_r = np.where(st == 1, v[..., 2], v[..., 3])   # pent p -> step p+1
    vb_r = np.where(st == 1, v[..., 3], v[..., 2])   # pent p -> step p+2
    va_h = np.where(st == 1, v[..., 4], v[..., 5])
    vb_h = np.where(st == 1, v[..., 5], v[..., 4])
    roll = np.empty((n, k - 3), dtype=np.float64)
    helt = np.empty((n, k - 3), dtype=np.float64)
    roll[:, 0] = va_r[:, 0]
    roll[:, -1] = vb_r[:, -1]
    helt[:, 0] = va_h[:, 0]
    helt[:, -1] = vb_h[:, -1]
    if k - 3 > 2:
        roll[:, 1:-1] = (va_r[:, 1:] + vb_r[:, :-1]) / 2.0
        helt[:, 1:-1] = (va_h[:, 1:] + vb_h[:, :-1]) / 2.0
    # getShape prints with setprecision(2); replicate its rounding exactly
    # (only the averaged interior step values can differ from 2-decimal
    # table values, but the endpoints are formatted too for idempotence).
    roll = _round2(roll)
    helt = _round2(helt)
    if single:
        return mgw[0], prot[0], roll[0], helt[0]
    return mgw, prot, roll, helt


def shape_stats_from_codes(codes):
    """(n, k) ACGT code matrix -> (n, 8) shape features: mean and SD
    (ddof=0, as the previous pipeline) of MGW, ProT, Roll, HelT across the
    sequence's valid positions/steps. Feature order:
    MGW_mean, MGW_std, PropTw_mean, PropTw_std, Roll_mean, Roll_std,
    HelTw_mean, HelTw_std."""
    mgw, prot, roll, helt = shape_arrays_from_codes(np.asarray(codes))
    out = np.empty((codes.shape[0], 8), dtype=np.float64)
    for j, arr in enumerate((mgw, prot, roll, helt)):
        out[:, 2 * j] = arr.mean(axis=1)
        out[:, 2 * j + 1] = arr.std(axis=1)
    return out


def dnashape_position_values(seq):
    """Per-position DNAshape values for a sequence string (see
    shape_arrays_from_codes). Sequences must be ACGT-only, length >= 5."""
    s = seq.upper()
    codes = np.array(["ACGT".index(b) for b in s], dtype=np.int64)
    return shape_arrays_from_codes(codes)


def dnashape_stats(seq):
    """8 DNA-shape features of a k-mer from the published pentamer query
    table (mean and SD, ddof=0, of MGW/ProT/Roll/HelT across the valid
    positions/steps). Same slot structure as the retired dinucleotide
    table. Returns zeros for non-ACGT or too-short sequences (mirrors the
    previous empty-fallback behaviour)."""
    s = seq.upper()
    if len(s) < 5 or any(c not in "ACGT" for c in s):
        return [0.0] * 8
    return shape_stats_from_codes(
        np.array(["ACGT".index(b) for b in s], dtype=np.int64)[None, :])[0].tolist()


def dinuc_shape_stats(seq):
    """DEPRECATED alias kept for API compatibility (v1.3.0): the old
    16-dinucleotide table was replaced by the published DNAshape pentamer
    query table; this now returns dnashape_stats(seq)."""
    return dnashape_stats(seq)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ------------------------------
# Promoter window helper + gene annotation
# ------------------------------
def load_promoters(gff_path, up=300, down=50):
    cds = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            contig, source, ftype, start, end, score, strand, phase, attr = parts
            if ftype != "CDS":
                continue
            start, end = int(start), int(end)
            gene_id = None
            for tag in attr.split(";"):
                if tag.startswith("ID=") or tag.startswith("Name=") or tag.startswith("gene="):
                    gene_id = tag.split("=")[-1]
                    break
            if strand == "+":
                tss = start
                prom_start = max(1, tss - up)
                prom_end = tss + down
            else:
                tss = end
                prom_start = max(1, tss - down)
                prom_end = tss + up
            cds.append((contig, prom_start, prom_end, strand, gene_id))
    return pd.DataFrame(cds, columns=["contig", "prom_start", "prom_end", "strand", "gene_id"])


def load_gene_names(gff_path):
    """Load gene/CDS names from a GFF3 file.

    Returns a list of (contig, start, end, gene_name) tuples.
    """
    genes = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            contig, ftype = parts[0], parts[2]
            if ftype != "gene" and ftype != "CDS":
                continue
            start, end, attr = int(parts[3]), int(parts[4]), parts[8]
            gene_id = None
            gene_name = None
            for field in attr.split(";"):
                if field.startswith("ID="):
                    gene_id = field.replace("ID=", "")
                elif field.startswith("Name="):
                    gene_name = field.replace("Name=", "")
                elif field.startswith("gene="):
                    gene_name = field.replace("gene=", "")
            if not gene_name and gene_id:
                gene_name = gene_id
            if gene_name:
                genes.append((contig, start, end, gene_name))
    return genes


def annotate_nearest_gene(hits, genes):
    """Annotate each hit (dict with contig/start/end keys) with the nearest
    gene ON THE SAME CONTIG, adding a 'nearest_gene' key in place."""
    by_contig = {}
    for contig, start, end, name in genes:
        by_contig.setdefault(contig, []).append((start, end, name))
    for hit in hits:
        sub = by_contig.get(hit["contig"])
        if not sub:
            hit["nearest_gene"] = "NA"
            continue
        mid = (hit["start"] + hit["end"]) // 2
        hit["nearest_gene"] = min(sub, key=lambda g: min(abs(mid - g[0]), abs(mid - g[1])))[2]
    return hits

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def read_motif(motif_path):
    """Read motif from JASPAR or MEME format with URL handling"""
    with open(motif_path, 'r') as f:
        content = f.read()

    # Remove URL lines that cause parsing issues
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.startswith('URL '):
            cleaned_lines.append(line)
    cleaned_content = '\n'.join(cleaned_lines)

    from io import StringIO
    for fmt in ("jaspar", "meme"):
        try:
            return motifs.read(StringIO(cleaned_content), fmt)
        except Exception:
            continue

    # If standard parsing fails, try to fix ALL spacing issues
    lines = cleaned_content.split('\n')
    fixed_lines = []
    for line in lines:
        if 'letter-probability matrix' in line:
            # Fix ALL spacing issues in the header
            line = line.replace('alength= ', 'alength=')
            line = line.replace('w=  ', 'w=')
            line = line.replace('w= ', 'w=')
            line = line.replace('nsites= ', 'nsites=')
            # Also handle cases with multiple spaces
            line = line.replace('  ', ' ')  # Replace double spaces with single
            line = line.replace('= ', '=')  # Remove spaces after equals
        fixed_lines.append(line)

    fixed_content = '\n'.join(fixed_lines)

    # Try parsing with fixed spacing
    for fmt in ("meme", "jaspar"):
        try:
            return motifs.read(StringIO(fixed_content), fmt)
        except Exception:
            continue

    # Final attempt: try manual parsing as last resort
    try:
        return parse_motif_manually(cleaned_content)
    except Exception:
        raise ValueError("Could not parse motif file with any method")

def parse_motif_manually(content):
    """Manual parsing as fallback for problematic MEME files"""
    lines = content.split('\n')
    matrix = []
    in_matrix = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if 'letter-probability matrix' in line:
            in_matrix = True
            continue

        if in_matrix and line and not line.startswith(('URL', 'MOTIF', 'ALPHABET', 'MEME')):
            # Parse probability values
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 4:  # ACGT probabilities
                    matrix.append(values)
            except ValueError:
                continue

    if matrix:
        # Create motif object manually
        from Bio.motifs import Motif

        # Convert probabilities to integer counts
        counts_array = (np.array(matrix) * 100).astype(int)

        # Create counts dictionary in the correct format
        counts_dict = {
            'A': counts_array[:, 0],  # A counts for each position
            'C': counts_array[:, 1],  # C counts for each position
            'G': counts_array[:, 2],  # G counts for each position
            'T': counts_array[:, 3]   # T counts for each position
        }

        # Create motif using counts dictionary
        motif = Motif(counts=counts_dict)
        return motif

    raise ValueError("Manual parsing failed - no valid matrix data found")

def pfm_to_pwm(motif_obj, pseudocount=0.5):
    """Convert a PFM (position frequency matrix) to per-column probabilities
    using a standard COUNT-BASED pseudocount.

    For each column:  p_b = (count_b + pc) / (N + 4 * pc)
    where N is the total number of aligned sites contributing to that column
    and pc is the pseudocount (default 0.5). This does NOT flatten confident
    columns: count-only columns keep probabilities close to the observed
    frequencies. Log-odds scores downstream use the natural logarithm
    (see logodds_score).
    """
    counts = motif_obj.counts
    L = motif_obj.length
    pwm = []
    for i in range(L):
        col = {b: float(counts[b][i]) if b in counts else 0.0 for b in "ACGT"}
        n_sites = sum(col.values())
        if n_sites <= 0:
            # No information for this column: use uniform probabilities.
            pwm.append({b: 0.25 for b in "ACGT"})
            continue
        pwm.append({b: (col[b] + pseudocount) / (n_sites + 4.0 * pseudocount) for b in col})
    return pwm

def genome_bg_freq(fasta_path):
    cnt = Counter(); total=0
    for rec in SeqIO.parse(fasta_path,"fasta"):
        s=str(rec.seq).upper()
        for b in s:
            if b in "ACGT":
                cnt[b]+=1
                total+=1
    return {b: cnt[b]/total for b in "ACGT"}

def logodds_score(seq, pwm, bg):
    """PWM log-odds score of a kmer against background frequencies.

    Uses the NATURAL logarithm consistently:  sum_i ln(p_i(b) / bg(b)).
    Returns None if the sequence contains non-ACGT characters.
    """
    s = 0.0; seq=seq.upper()
    for i,b in enumerate(seq):
        if b not in "ACGT": return None
        s += math.log(pwm[i].get(b,1e-9)/bg.get(b,1e-9))
    return s

def seq_entropy(seq):
    s = seq.upper()
    cnt = Counter(s)
    probs = [cnt[b]/len(s) for b in "ACGT" if cnt[b] > 0]
    if not probs: return 0.0
    arr = np.array(probs,dtype=float)
    return float(-np.sum(arr*np.log2(arr)))

def merge_hits(hits):
    if not hits: return []
    sorted_hits = sorted(hits,key=lambda h:(h['contig'],h['start'],-h['score']))
    merged = []; cur=sorted_hits[0].copy()
    for h in sorted_hits[1:]:
        if h['contig']!=cur['contig'] or h['start']>cur['end']:
            merged.append(cur); cur=h.copy()
        else:
            cur['end']=max(cur['end'],h['end'])
            if h['score']>cur['score']: cur.update({k:h[k] for k in ('start','end','score','seq','strand')})
    merged.append(cur); return merged

def process_batch_predict(seqs_chunk, pos_chunk, strand_chunk, motif_len, pwm, bg, clf, prob_cutoff, contig_id):
    rows=[]; ok_idxs=[]
    for idx,(s,strand) in enumerate(seqs_chunk):
        llr=logodds_score(s,pwm,bg)
        if llr is None: rows.append([np.nan]*11); continue
        gc=(s.count('G')+s.count('C'))/len(s)
        ent=seq_entropy(s)
        shape=dinuc_shape_stats(s)
        rows.append([llr,gc,ent]+shape); ok_idxs.append(idx)
    Xb=np.array(rows,dtype=float); ok=~np.isnan(Xb).any(axis=1)
    hits_local=[]
    if ok.sum()>0:
        probs=clf.predict_proba(Xb[ok])[:,1]
        ok_positions=[i for i,f in enumerate(ok) if f]
        k=0
        for pos_idx in ok_positions:
            p=float(probs[k])
            if p>=float(prob_cutoff):
                seq,strand=seqs_chunk[pos_idx]
                start=pos_chunk[pos_idx]
                if strand == '-':
                    end = start
                    start = start - motif_len + 1
                else:
                    end = start + motif_len - 1
                hits_local.append({'contig':contig_id,'start':start,'end':end,'strand':strand,'score':p,'seq':seq})
            k+=1
    return hits_local

def get_both_strands(kmer):
    try:
        rev_comp = str(BSeq(kmer).reverse_complement())
        return [(kmer, '+'), (rev_comp, '-')]
    except Exception:
        return [(kmer, '+')]

# ---------------------------
# SCAN WINDOW / NEGATIVE SAMPLING HELPERS
# ---------------------------
_BASE_A, _BASE_C, _BASE_G, _BASE_T = (65, 67, 71, 84)  # ASCII codes for ACGT

def valid_window_starts(region_seq, occ_region, motif_len):
    """Return 0-based offsets (relative to region_seq) of all motif_len windows
    that (a) consist only of ACGT and (b) do not overlap occupied positions."""
    n = len(region_seq) - motif_len + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(region_seq.encode("ascii"), dtype=np.uint8)
    valid_base = np.isin(arr, (_BASE_A, _BASE_C, _BASE_G, _BASE_T))
    invalid = (~valid_base) | occ_region
    cs = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
    win = cs[motif_len:motif_len + n] - cs[:n]
    return np.nonzero(win == 0)[0]


def iter_scan_batches(scan_regions, contig_seqs, motif_len, batch_size):
    """Yield (seqs_chunk, positions, strands, contig_id) batches over all scan
    regions (contig, start1, end1_inclusive). Both strands of every valid
    window are emitted. Batches never mix contigs."""
    seqs_chunk, positions, strands = [], [], []
    cur_contig = None
    for contig, rstart, rend in scan_regions:
        if contig not in contig_seqs:
            continue
        if cur_contig is not None and contig != cur_contig and seqs_chunk:
            yield seqs_chunk, positions, strands, cur_contig
            seqs_chunk, positions, strands = [], [], []
        cur_contig = contig
        region_seq = contig_seqs[contig][rstart - 1:rend]
        for i in range(0, len(region_seq) - motif_len + 1):
            kmer = region_seq[i:i + motif_len]
            if any(c not in "ACGT" for c in kmer):
                continue
            for strand_kmer, strand in get_both_strands(kmer):
                seqs_chunk.append((strand_kmer, strand))
                positions.append(rstart + i)
                strands.append(strand)
            if len(seqs_chunk) >= batch_size:
                yield seqs_chunk, positions, strands, contig
                seqs_chunk, positions, strands = [], [], []
    if seqs_chunk:
        yield seqs_chunk, positions, strands, cur_contig


def build_occupied(contig_seqs, pos_meta, positives, pad, motif_len):
    """Boolean per-contig arrays marking +/-pad around known positive sites.

    Positive locations come from (a) CollecTF coordinates (pos_meta), or
    (b) exact matches of the provided known sequences (and their reverse
    complements) when only plain sequences are available via --known.
    """
    occupied = {cid: np.zeros(len(seq), dtype=bool) for cid, seq in contig_seqs.items()}
    for m in pos_meta:
        cid = m['contig']
        if cid not in occupied:
            continue
        a = max(0, int(m['start']) - 1 - pad)
        b = min(len(occupied[cid]), int(m['end']) + pad)
        occupied[cid][a:b] = True
    if not pos_meta:
        for cid, seq in contig_seqs.items():
            occ = occupied[cid]
            for s in set(positives):
                for query in (s, str(BSeq(s).reverse_complement())):
                    start = seq.find(query)
                    while start != -1:
                        a = max(0, start - pad)
                        b = min(len(seq), start + len(query) + pad)
                        occ[a:b] = True
                        start = seq.find(query, start + 1)
    return occupied


def sample_negatives(scan_regions, contig_seqs, occupied, motif_len, n_neg, rng):
    """Sample negative windows from the SAME windows that will be scanned
    (excluding +/-pad around positive sites). Both strands of each sampled
    window are used. Deterministic for a fixed seeded rng."""
    cand_contig_idx = []
    cand_pos = []
    for ci, (contig, rstart, rend) in enumerate(scan_regions):
        if contig not in contig_seqs:
            continue
        region_seq = contig_seqs[contig][rstart - 1:rend]
        occ_region = occupied[contig][rstart - 1:rend]
        starts = valid_window_starts(region_seq, occ_region, motif_len)
        if len(starts) == 0:
            continue
        cand_contig_idx.append(np.full(len(starts), ci, dtype=np.int64))
        cand_pos.append(starts + (rstart - 1))  # convert to 0-based contig coords
    if not cand_pos:
        return []
    flat_ci = np.concatenate(cand_contig_idx)
    flat_pos = np.concatenate(cand_pos)
    n_available = len(flat_pos)
    if n_available < n_neg:
        logging.warning(
            f"Available negative windows ({n_available}) fewer than requested "
            f"({n_neg}). Reducing n_neg accordingly."
        )
        n_neg = n_available
    sel = rng.choice(n_available, size=n_neg, replace=False) if n_available > n_neg else np.arange(n_available)
    negatives = []
    for j in sel:
        contig, rstart, _ = scan_regions[flat_ci[j]]
        p = int(flat_pos[j])
        kmer = contig_seqs[contig][p:p + motif_len]
        for strand_kmer, strand in get_both_strands(kmer):
            negatives.append(strand_kmer)
            if len(negatives) >= n_neg:
                break
        if len(negatives) >= n_neg:
            break
    return negatives

# ---------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------
def run(args):
    # Deterministic runs: seed python's random and NumPy RNGs.
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
    if not fasta_records:
        raise SystemExit("No FASTA records found.")
    logging.info(f"Loaded {len(fasta_records)} contigs from FASTA.")
    contig_seqs = {rec.id: str(rec.seq).upper() for rec in fasta_records}

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    motif = read_motif(args.motif)
    motif_len = motif.length
    pwm = pfm_to_pwm(motif, pseudocount=float(args.pseudocount))
    logging.info(f"Motif length: {motif_len}; PWM built with count-based pseudocount={args.pseudocount} (natural-log log-odds)")

    bg = genome_bg_freq(args.fasta)
    logging.info(f"Background: {bg}")

    pad = int(args.pad)
    positives = []
    pos_meta = []

    # Positive windows from the bundled CollecTF export. --genome_accession is
    # used ONLY for looking up known sites here; scanning covers ALL contigs.
    logging.info(f"Checking known dataset at {KNOWN_DATASET_PATH}")
    try:
        known_df = pd.read_csv(KNOWN_DATASET_PATH, sep='\t')

        required_cols = ['TF', 'genome_accession', 'site_start', 'site_end', 'site_strand']
        missing_cols = [col for col in required_cols if col not in known_df.columns]

        if missing_cols:
            logging.warning(f"Known dataset missing columns: {missing_cols}. Skipping known dataset.")
        else:
            pos_df = known_df[
                (known_df['TF'] == args.gene) &
                (known_df['genome_accession'] == args.genome_accession)
            ].copy()

            logging.info(f"Found {pos_df.shape[0]} known sites for TF {args.gene} on {args.genome_accession}.")

            if not pos_df.empty:
                contig_seq_for_known = contig_seqs.get(args.genome_accession)
                if contig_seq_for_known is None:
                    logging.warning(f"Genome accession {args.genome_accession} not found in FASTA for known site extraction")
                else:
                    # Extract best matching kmer from padded regions
                    for _, r in pos_df.iterrows():
                        try:
                            s = int(r['site_start']); e = int(r['site_end'])
                        except Exception:
                            continue
                        a = max(1, s - pad); b = min(len(contig_seq_for_known), e + pad)
                        region = contig_seq_for_known[a-1:b]

                        best = None; best_kmer = None; best_pos = None; best_strand = None
                        for i in range(0, len(region) - motif_len + 1):
                            for strand_kmer, strand in get_both_strands(region[i:i+motif_len]):
                                sc = logodds_score(strand_kmer, pwm, bg)
                                if sc is None:
                                    continue
                                if best is None or sc > best:
                                    best = sc
                                    best_kmer = strand_kmer
                                    best_pos = a + i
                                    best_strand = strand

                        if best_kmer:
                            positives.append(best_kmer)
                            pos_meta.append({
                                'contig': args.genome_accession,
                                'start': best_pos,
                                'end': best_pos + motif_len - 1,
                                'strand': best_strand
                            })

                    logging.info(f"Collected {len(positives)} positive windows (best kmer ±{pad}bp).")
            else:
                logging.info(f"No known binding sites found for {args.gene} in {args.genome_accession}")

    except Exception as e:
        logging.warning(f"Error reading known dataset: {e}")
        logging.warning("Please check the file format and column names")
        positives = []
        pos_meta = []

    # Fallback: plain-text known binding sequences via --known.
    if not positives and args.known:
        if not os.path.exists(args.known):
            raise SystemExit(f"Known dataset file not found at {args.known}.")
        with open(args.known) as f:
            for line in f:
                seq = line.strip().upper()
                if seq and all(c in "ACGT" for c in seq):
                    positives.append(seq)
        logging.info(f"Loaded {len(positives)} known binding sequences from {args.known}.")

    if not positives:
        raise SystemExit(
            "No known binding sites available for supervised training. "
            "Provide them with --known (plain txt, one sequence per line) or use "
            "a --gene/--genome_accession pair present in the bundled CollecTF export."
        )

    # ---------------------------
    # Define the windows that will be scanned
    # ---------------------------
    promoters = None
    if args.gff:
        promoters = load_promoters(args.gff, up=int(args.up), down=int(args.down))

    if args.full_genome:
        # Full-genome mode: slide windows across the ENTIRE length of every contig.
        scan_regions = [(rec.id, 1, len(contig_seqs[rec.id])) for rec in fasta_records]
        logging.info(f"Full-genome scanning enabled: sliding windows across all {len(scan_regions)} contigs.")
    else:
        if promoters is None:
            raise SystemExit("Error: --gff is required for promoter-based scanning")
        scan_regions = [
            (r.contig, int(r.prom_start), int(r.prom_end))
            for r in promoters.itertuples()
            if r.contig in contig_seqs
        ]
        logging.info(f"Promoter-based scanning: {len(scan_regions)} promoter windows across all contigs.")

    # ---------------------------
    # Negative sampling from the SAME windows that will be scanned
    # ---------------------------
    n_neg_requested = max(int(args.neg_ratio * len(positives)), int(args.min_negatives))
    occupied = build_occupied(contig_seqs, pos_meta, positives, pad, motif_len)
    negatives = sample_negatives(scan_regions, contig_seqs, occupied, motif_len, n_neg_requested, rng)
    if not negatives:
        raise SystemExit("Could not sample any negative windows from the scan regions.")
    logging.info(
        f"Training set: {len(positives)} positives, {len(negatives)} negatives "
        f"(requested {n_neg_requested}; neg_ratio={args.neg_ratio}, min_negatives={args.min_negatives}, seed={seed})."
    )

    X, y = [], []
    for s in positives:
        X.append([logodds_score(s, pwm, bg),
                  (s.count('G') + s.count('C')) / len(s),
                  seq_entropy(s)] + dinuc_shape_stats(s))
        y.append(1)
    for s in negatives:
        X.append([logodds_score(s, pwm, bg),
                  (s.count('G') + s.count('C')) / len(s),
                  seq_entropy(s)] + dinuc_shape_stats(s))
        y.append(0)
    X = np.array(X)
    y = np.array(y, dtype=int)

    logging.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=int(args.n_trees), n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    logging.info("Training completed.")

    if args.auto_cutoff:
        logging.info("Selecting optimal probability cutoff via PR curve (maximize F1)...")
        probs_all = clf.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, probs_all)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_cutoff = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logging.info(f"Optimal probability cutoff: {best_cutoff:.4f}, F1={f1_scores[best_idx]:.4f}")
        args.prob_cutoff = best_cutoff

    # ---------------------------
    # Parallel batch scanning of ALL contigs
    # ---------------------------
    batch_size = int(args.batch)
    batches = iter_scan_batches(scan_regions, contig_seqs, motif_len, batch_size)
    results = Parallel(n_jobs=-1)(
        delayed(process_batch_predict)(
            seqs_chunk, positions, strands, motif_len,
            pwm, bg, clf, args.prob_cutoff, contig_id
        )
        for seqs_chunk, positions, strands, contig_id in tqdm(batches, desc="Scanning batches", unit="batch")
    )
    hits = [h for batch_hits in results for h in batch_hits]

    logging.info(f"Scanning completed: {len(hits)} candidate sites predicted.")

    hits_merged = merge_hits(hits)

    top_fraction = 0.05
    hits_merged_sorted = sorted(hits_merged, key=lambda x: x['score'], reverse=True)
    n_top = max(1, int(len(hits_merged_sorted) * top_fraction)) if hits_merged_sorted else 0
    top_hits = hits_merged_sorted[:n_top]

    # Nearest-gene annotation (contig-aware). hits_merged, hits_merged_sorted
    # and top_hits share the same dict objects, so a single pass annotates all.
    genes = load_gene_names(args.gff) if args.gff else []
    annotate_nearest_gene(hits_merged_sorted, genes)

    df_out_full = pd.DataFrame(hits_merged_sorted)
    out_path_full = os.path.join(outdir, f"{args.gene}_predictions_full.csv")
    df_out_full.to_csv(out_path_full, index=False)
    logging.info(f"Full predictions saved to {out_path_full} ({len(hits_merged_sorted)} hits)")

    df_out_top = pd.DataFrame(top_hits)
    out_path_top = os.path.join(outdir, f"{args.gene}_predictions_top{int(top_fraction * 100)}pct.csv")
    df_out_top.to_csv(out_path_top, index=False)
    logging.info(f"Top {int(top_fraction * 100)}% high-confidence predictions saved to {out_path_top} ({len(top_hits)} hits)")

    logging.info(f"Merged overlapping hits: {len(hits_merged)} final candidate sites.")

    df_out = pd.DataFrame(hits_merged)
    out_path = os.path.join(outdir, f"{args.gene}_predictions.csv")
    df_out.to_csv(out_path, index=False)
    logging.info(f"Predictions saved to {out_path} with gene annotation")

    # Per-contig hit counts for logging
    if hits_merged:
        contig_counts = Counter(h['contig'] for h in hits_merged)
        for cid in sorted(contig_counts):
            logging.info(f"  {cid}: {contig_counts[cid]} predictions")

    # ==============
    # VISUALIZATIONS 
    # ==============

    # 1. FEATURE IMPORTANCE PLOT
    try:
        feature_names = ['PWM_score', 'GC_content', 'Entropy', 
                        'MGW_mean', 'MGW_std', 'PropTw_mean', 'PropTw_std',
                        'Roll_mean', 'Roll_std', 'HelTw_mean', 'HelTw_std']
        
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices], align="center", alpha=0.7)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance for {args.gene} TFBS Prediction')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        importance_path = os.path.join(outdir, f"{args.gene}_feature_importance.png")
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Feature importance plot saved to {importance_path}")
    except Exception as e:
        logging.warning(f"Could not generate feature importance plot: {e}")

    # 2. GENOME DISTRIBUTION PLOT (contig-aware: cumulative offsets per contig)
    if hits_merged:
        try:
            contig_offsets = {}
            offset = 0
            for rec in fasta_records:
                contig_offsets[rec.id] = offset
                offset += len(contig_seqs[rec.id])

            plt.figure(figsize=(14, 6))
            positions = [(hit['start'] + hit['end']) // 2 + contig_offsets[hit['contig']] for hit in hits_merged]
            scores = [hit['score'] for hit in hits_merged]

            plt.scatter(positions, scores, alpha=0.6, s=30, c=scores, cmap='viridis')
            plt.colorbar(label='Prediction Score')
            # Mark contig boundaries
            for cid, off in contig_offsets.items():
                plt.axvline(x=off, color='gray', linestyle=':', alpha=0.5)
                plt.text(off, plt.ylim()[1], f' {cid}', rotation=90, va='top', fontsize=8, color='gray')
            plt.xlabel('Genomic Position (contigs concatenated)')
            plt.ylabel('Prediction Score')
            plt.title(f'TFBS Predictions along Genome for {args.gene}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            distribution_path = os.path.join(outdir, f"{args.gene}_genome_distribution.png")
            plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Genome distribution plot saved to {distribution_path}")
        except Exception as e:
            logging.warning(f"Could not generate genome distribution plot: {e}")

    # 3. SCORE DISTRIBUTION HISTOGRAM
    if hits_merged:
        try:
            plt.figure(figsize=(10, 6))
            scores = [hit['score'] for hit in hits_merged]
            
            plt.hist(scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
            plt.xlabel('Prediction Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Prediction Scores for {args.gene}')
            plt.grid(True, alpha=0.3)
            
            plt.axvline(x=args.prob_cutoff, color='red', linestyle='--', 
                        linewidth=2, label=f'Cutoff: {args.prob_cutoff:.3f}')
            plt.legend()
            
            plt.tight_layout()
            hist_path = os.path.join(outdir, f"{args.gene}_score_distribution.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Score distribution histogram saved to {hist_path}")
        except Exception as e:
            logging.warning(f"Could not generate score distribution plot: {e}")

    # 4. PROMOTER CONTEXT PLOT (distances computed within the same contig)
    if promoters is not None and hits_merged:
        try:
            tss_by_contig = {}
            for row in promoters.itertuples():
                if row.strand == "+":
                    tss = row.prom_start + int(args.up)
                else:
                    tss = row.prom_end - int(args.down)
                tss_by_contig.setdefault(row.contig, []).append(tss)

            if tss_by_contig:
                plt.figure(figsize=(12, 6))

                distances = []
                for hit in hits_merged:
                    tss_list = tss_by_contig.get(hit['contig'])
                    if not tss_list:
                        continue
                    hit_mid = (hit['start'] + hit['end']) // 2
                    distances.append(min(abs(hit_mid - tss) for tss in tss_list))

                if distances:
                    # Plot histogram of distances
                    plt.hist(distances, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
                    plt.xlabel('Distance to Nearest TSS (bp)')
                    plt.ylabel('Number of Predictions')
                    plt.title(f'TFBS Distribution Relative to Transcription Start Sites for {args.gene}')
                    plt.grid(True, alpha=0.3)

                    # Add vertical lines for common promoter regions
                    plt.axvline(x=0, color='red', linestyle='-', alpha=0.7, label='TSS')
                    plt.axvline(x=-50, color='orange', linestyle='--', alpha=0.7, label='-50 bp')
                    plt.axvline(x=-100, color='orange', linestyle='--', alpha=0.7, label='-100 bp')
                    plt.axvline(x=+50, color='blue', linestyle='--', alpha=0.7, label='+50 bp')

                    plt.legend()
                    plt.tight_layout()
                    promoter_path = os.path.join(outdir, f"{args.gene}_promoter_context.png")
                    plt.savefig(promoter_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logging.info(f"Promoter context plot saved to {promoter_path}")
        except Exception as e:
            logging.warning(f"Could not generate promoter context plot: {e}")

    logging.info("All visualizations completed successfully!")
