#!/usr/bin/env python3
"""
D-Sites: Hybrid TFBS predictor combining PWM, dinucleotide shape, and Random Forest

Author: Pankaj
Institution: IARI
Contact: ft.pank@gmail.com
License: MIT

Version: 1.1.0
Date: 2025

Description:
-------------
This tool predicts transcription factor binding sites (TFBS) in bacterial genomes
using a hybrid computational framework that combines:

1. Position Weight Matrix (PWM) scoring with pseudocount adjustment
2. Dinucleotide DNA-shape features (MGW, Propeller Twist, Roll, Helical Twist)
3. Sequence-derived features (GC content, sequence entropy)
4. Random Forest classifier for supervised prediction

The pipeline performs genome-wide scanning in a memory-efficient manner and
allows automatic probability cutoff selection via precision-recall optimization.

Publication Notes:
-------------------
- TFBS prediction combines sequence-specific and DNA structural features for improved accuracy.
- Optimized negative sampling reduces bias and improves classifier training.
- Batch-based genome scanning enables large genome analysis without memory overhead.
- The tool provides output as CSV with predicted binding sites, strand orientation, and probability score.

References:
-----------
- DNA shape feature parameters from published dinucleotide models.
- Random Forest supervised classifier for TFBS prediction.
- PWM scoring methodology with pseudocount adjustment.

"""

import os
import sys
import math
import argparse
import random
import itertools
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq as BSeq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm


__version__ = "1.1.0"

# Required package versions for reproducibility
REQUIRED_PACKAGES = {
    'python': '>=3.8',
    'biopython': '>=1.79',
    'scikit-learn': '>=1.0',
    'pandas': '>=1.3',
    'numpy': '>=1.21',
    'joblib': '>=1.1',
    'tqdm': '>=4.62'
}

# ---------------------------
# HARD-CODED known dataset path
# ---------------------------
KNOWN_DATASET_PATH = "./collectf_export.tsv"

# ---------------------------
# DINUCLEOTIDE PARAMETERS
# ---------------------------
DINUC_PARAMS = {
    'AA': {'mgw': 4.0, 'prop_tw': -14.0, 'roll': 0.6, 'helix_twist': 34.3},
    'AT': {'mgw': 4.2, 'prop_tw': -13.3, 'roll': 1.1, 'helix_twist': 34.1},
    'AC': {'mgw': 4.1, 'prop_tw': -14.5, 'roll': 0.9, 'helix_twist': 34.6},
    'AG': {'mgw': 4.0, 'prop_tw': -14.0, 'roll': 1.0, 'helix_twist': 34.4},
    'TA': {'mgw': 4.7, 'prop_tw': -11.8, 'roll': 3.3, 'helix_twist': 33.9},
    'TT': {'mgw': 4.0, 'prop_tw': -14.0, 'roll': 0.6, 'helix_twist': 34.3},
    'TC': {'mgw': 4.1, 'prop_tw': -14.8, 'roll': 0.7, 'helix_twist': 34.7},
    'TG': {'mgw': 4.2, 'prop_tw': -14.3, 'roll': 1.2, 'helix_twist': 34.2},
    'CA': {'mgw': 4.1, 'prop_tw': -14.8, 'roll': 0.7, 'helix_twist': 34.5},
    'CT': {'mgw': 4.1, 'prop_tw': -14.5, 'roll': 0.9, 'helix_twist': 34.4},
    'CC': {'mgw': 4.2, 'prop_tw': -15.0, 'roll': 0.8, 'helix_twist': 34.8},
    'CG': {'mgw': 4.3, 'prop_tw': -14.8, 'roll': 1.3, 'helix_twist': 34.9},
    'GA': {'mgw': 4.0, 'prop_tw': -14.0, 'roll': 1.0, 'helix_twist': 34.2},
    'GT': {'mgw': 4.2, 'prop_tw': -14.3, 'roll': 1.2, 'helix_twist': 34.2},
    'GC': {'mgw': 4.3, 'prop_tw': -14.8, 'roll': 1.3, 'helix_twist': 34.7},
    'GG': {'mgw': 4.2, 'prop_tw': -14.5, 'roll': 1.1, 'helix_twist': 34.5}
}

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


def annotate_hits_with_genes(hits_df, gff_path):
    """Annotate predicted sites with nearest gene from GFF"""
    genes = []
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
            genes.append((contig, start, end, gene_id))
    
    gene_df = pd.DataFrame(genes, columns=["contig", "start", "end", "gene_id"])
    annotations = []
    
    for _, row in hits_df.iterrows():
        sub = gene_df[gene_df["contig"] == row["contig"]]
        if sub.empty:
            annotations.append(None)
            continue
        dists = ((sub["start"] + sub["end"]) // 2 - ((row["start"] + row["end"]) // 2)).abs()
        nearest_idx = dists.idxmin()
        annotations.append(sub.loc[nearest_idx, "gene_id"])
    
    hits_df["nearest_gene"] = annotations
    return hits_df

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
        except Exception as e:
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
        except Exception as e:
            continue
    
    # Final attempt: try manual parsing as last resort
    try:
        return parse_motif_manually(cleaned_content)
    except:
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
        # Create motif object manually - FIXED
        from Bio.motifs import Motif
        import numpy as np
        
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
    counts = motif_obj.counts
    L = motif_obj.length
    pwm = []
    for i in range(L):
        col = {b: float(counts[b][i]) if b in counts else 0.0 for b in "ACGT"}
        s = sum(col.values())
        if s <= 0:
            col = {b:1.0 for b in "ACGT"}
            s = 4.0
        coln = {b: (col[b]/s + pseudocount)/(1 + 4*pseudocount) for b in col}
        pwm.append(coln)
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

def dinuc_shape_stats(seq):
    s=seq.upper(); mgw=[]; prot=[]; roll=[]; helt=[]
    for i in range(len(s)-1):
        din = s[i:i+2]
        if any(ch not in "ACGT" for ch in din): continue
        p = DINUC_PARAMS.get(din) or DINUC_PARAMS.get(str(BSeq(din).reverse_complement()))
        if not p: continue
        mgw.append(p['mgw']); prot.append(p['prop_tw']); roll.append(p['roll']); helt.append(p['helix_twist'])
    def st(x):
        if not x: return 0.0,0.0
        arr=np.array(x,dtype=float)
        return float(arr.mean()),float(arr.std())
    out=[]
    for arr in (mgw,prot,roll,helt): m,sigma=st(arr); out.extend([m,sigma])
    return out

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
    except:
        return [(kmer, '+')]

# ---------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------
def run(args):
    random.seed(args.seed)
    np.random.seed(int(args.seed))

    fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
    if not fasta_records:
        raise SystemExit("No FASTA records found.")
    logging.info(f"Loaded {len(fasta_records)} contigs from FASTA.")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    motif = read_motif(args.motif)
    motif_len = motif.length
    pwm = pfm_to_pwm(motif)
    logging.info(f"Motif length: {motif_len}")

    bg = genome_bg_freq(args.fasta)
    logging.info(f"Background: {bg}")

    positives = []
    pos_meta = []
    supervised = True

    # MODIFIED SECTION: Use the same logic as test.py for positive sequence extraction
    if os.path.exists(KNOWN_DATASET_PATH):
        logging.info(f"Checking known dataset at {KNOWN_DATASET_PATH}")
        try:
            # Read the TSV file
            known_df = pd.read_csv(KNOWN_DATASET_PATH, sep='\t')
            
            # Check if required columns exist
            required_cols = ['TF', 'genome_accession', 'site_start', 'site_end', 'site_strand']
            missing_cols = [col for col in required_cols if col not in known_df.columns]
            
            if missing_cols:
                logging.warning(f"Known dataset missing columns: {missing_cols}. Skipping known dataset.")
                positives = []
            else:
                # Filter for current TF and genome accession
                pos_df = known_df[
                    (known_df['TF'] == args.gene) & 
                    (known_df['genome_accession'] == args.genome_accession)
                ].copy()
                
                logging.info(f"Found {pos_df.shape[0]} known sites for TF {args.gene} on {args.genome_accession}.")
                
                if not pos_df.empty:
                    # Find the correct contig sequence first
                    contig_seq_for_known = None
                    for rec in fasta_records:
                        if rec.id == args.genome_accession:
                            contig_seq_for_known = str(rec.seq).upper()
                            break
                    
                    if contig_seq_for_known is None:
                        logging.warning(f"Genome accession {args.genome_accession} not found in FASTA for known site extraction")
                    else:
                        # Extract best matching kmer from padded regions (same as test.py)
                        pad = int(args.pad)
                        for _, r in pos_df.iterrows():
                            try: 
                                s = int(r['site_start']); e = int(r['site_end'])
                            except: 
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
    else:
        logging.info(f"Known dataset file not found at {KNOWN_DATASET_PATH}")

    # If no positives found in hardcoded file, check --known argument
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
            raise SystemExit("No valid sequences found in known dataset.")
    
    # Only prompt if no positives found in either source
    elif not positives:
        choice = input("No known dataset found. Do you want to (1) provide a path or (2) continue unsupervised? ").strip()
        if choice == "1":
            path = input("Enter path to known dataset (plain txt, one sequence per line): ").strip()
            if not os.path.exists(path):
                raise SystemExit(f"Known dataset file not found at {path}.")
            with open(path) as f:
                for line in f:
                    seq = line.strip().upper()
                    if seq and all(c in "ACGT" for c in seq):
                        positives.append(seq)
            logging.info(f"Loaded {len(positives)} known binding sequences from {path}.")
            if not positives:
                raise SystemExit("No valid sequences found in known dataset.")
        else:
            supervised = False
            logging.warning("Proceeding in UNSUPERVISED mode (PWM+shape only, no RF training).")

    pad = int(args.pad)
    contig_rec = None
    for rec in fasta_records:
        if rec.id == args.genome_accession:
            contig_rec = rec
            break
    if not contig_rec:
        raise SystemExit(f"Genome accession '{args.genome_accession}' not found in FASTA.")
    contig_seq = str(contig_rec.seq).upper()

    clf = None
    if supervised:
        n_neg = max(int(args.neg_ratio * len(positives)), 1000)
        
        # Mark regions around positive sites as occupied (same as test.py)
        occupied = np.zeros(len(contig_seq), dtype=bool)
        for m in pos_meta:
            a = max(0, int(m['start']) - 1 - pad)
            b = min(len(contig_seq), int(m['end']) + pad)
            occupied[a:b] = True
        
        available_pos = np.where(~occupied[:len(contig_seq) - motif_len])[0]
        if len(available_pos) < n_neg:
            n_neg = len(available_pos)
            logging.warning("Available negative positions less than requested. Reducing n_neg accordingly.")
        
        selected_neg = np.random.choice(available_pos, n_neg, replace=False)
        negatives = []
        for i in selected_neg:
            kmer = contig_seq[i:i + motif_len]
            if any(c not in "ACGT" for c in kmer):
                continue
            for strand_kmer, strand in get_both_strands(kmer):
                negatives.append(strand_kmer)
                if len(negatives) >= n_neg:
                    break
        logging.info(f"Collected {len(negatives)} negatives (requested {n_neg}).")

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
        clf = RandomForestClassifier(n_estimators=int(args.n_trees), n_jobs=-1, random_state=int(args.seed))
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
    else:
        logging.warning("Skipping RF training — using UNSUPERVISED PWM+shape scanning.")

    logging.info("Scanning only promoter windows for candidate TFBS...")

    if not args.gff:
        raise SystemExit("Error: --gff is required for promoter-based scanning")

    promoters = load_promoters(args.gff, up=int(args.up), down=int(args.down))
    hits = []
    batch_size = int(args.batch)

    for _, row in tqdm(promoters.iterrows(), total=promoters.shape[0]):
        prom_seq = contig_seq[row["prom_start"] - 1:row["prom_end"]]
        seqs_chunk, positions, strands = [], [], []
        for i in range(0, len(prom_seq) - motif_len + 1):
            kmer = prom_seq[i:i + motif_len]
            if any(c not in "ACGT" for c in kmer):
                continue
            for strand_kmer, strand in get_both_strands(kmer):
                seqs_chunk.append((strand_kmer, strand))
                positions.append(row["prom_start"] + i)
                strands.append(strand)
            if len(seqs_chunk) >= batch_size:
                if supervised:
                    hits.extend(process_batch_predict(
                        seqs_chunk, positions, strands, motif_len,
                        pwm, bg, clf, args.prob_cutoff, args.genome_accession
                    ))
                else:
                    for seq_idx, (seq, strand) in enumerate(seqs_chunk):
                        score = logodds_score(seq, pwm, bg)
                        if score is None:
                            continue
                        start = positions[seq_idx]
                        end = start + motif_len - 1 if strand == "+" else start
                        hits.append({'contig': args.genome_accession, 'start': start, 'end': end,
                                     'strand': strand, 'score': score, 'seq': seq})
                seqs_chunk, positions, strands = [], [], []
        if seqs_chunk:
            if supervised:
                hits.extend(process_batch_predict(
                    seqs_chunk, positions, strands, motif_len,
                    pwm, bg, clf, args.prob_cutoff, args.genome_accession
                ))
            else:
                for seq_idx, (seq, strand) in enumerate(seqs_chunk):
                    score = logodds_score(seq, pwm, bg)
                    if score is None:
                        continue
                    start = positions[seq_idx]
                    end = start + motif_len - 1 if strand == "+" else start
                    hits.append({'contig': args.genome_accession, 'start': start, 'end': end,
                                 'strand': strand, 'score': score, 'seq': seq})

    logging.info(f"Promoter scanning completed: {len(hits)} candidate sites predicted.")


    hits_merged = merge_hits(hits)

    top_fraction = 0.05
    hits_merged_sorted = sorted(hits_merged, key=lambda x: x['score'], reverse=True)
    n_top = max(1, int(len(hits_merged_sorted) * top_fraction))
    top_hits = hits_merged_sorted[:n_top]

    # ADD NEAREST GENE ANNOTATION TO BOTH FULL AND TOP HITS
    # Load genes from GFF for annotation
    genes = []
    if args.gff:
        with open(args.gff) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                ftype = parts[2]
                if ftype != "gene" and ftype != "CDS":  # Look for both gene and CDS features
                    continue
                start, end, strand, attr = int(parts[3]), int(parts[4]), parts[6], parts[8]
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
                if gene_name:  # Only add if we found a gene name
                    genes.append((start, end, gene_name))

    # Add nearest gene to all hits before creating DataFrames
    for hit in hits_merged_sorted:
        mid = (hit["start"] + hit["end"]) // 2
        hit["nearest_gene"] = min(genes, key=lambda g: min(abs(mid - g[0]), abs(mid - g[1])))[2] if genes else "NA"

    for hit in top_hits:
        mid = (hit["start"] + hit["end"]) // 2
        hit["nearest_gene"] = min(genes, key=lambda g: min(abs(mid - g[0]), abs(mid - g[1])))[2] if genes else "NA"


    df_out_full = pd.DataFrame(hits_merged_sorted)
    out_path_full = os.path.join(outdir, f"{args.gene}_predictions_full.csv")
    df_out_full.to_csv(out_path_full, index=False)
    if supervised:
        logging.info(f"Full predictions saved to {out_path_full} ({len(hits_merged_sorted)} hits)")
    else:
        logging.info(f"Full predictions saved to {out_path_full} ({len(hits_merged_sorted)} hits, UNSUPERVISED mode)")

    df_out_top = pd.DataFrame(top_hits)
    out_path_top = os.path.join(outdir, f"{args.gene}_predictions_top{int(top_fraction * 100)}pct.csv")
    df_out_top.to_csv(out_path_top, index=False)
    if supervised:
        logging.info(f"Top {int(top_fraction * 100)}% high-confidence predictions saved to {out_path_top} ({len(top_hits)} hits)")
    else:
        logging.info(f"Top {int(top_fraction * 100)}% high-confidence predictions saved to {out_path_top} ({len(top_hits)} hits, UNSUPERVISED mode)")

    # if supervised:
    #     probs_all = clf.predict_proba(X)[:, 1]
    #     precision, recall, thresholds = precision_recall_curve(y, probs_all)
    #     pr_auc = auc(recall, precision)

    #     plt.figure(figsize=(6, 6))
    #     plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC={pr_auc:.3f})')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title(f'Precision-Recall Curve for {args.gene}')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     pr_curve_path = os.path.join(outdir, f"{args.gene}_PR_curve.png")
    #     plt.savefig(pr_curve_path)
    #     plt.close()
    #     logging.info(f"Precision-Recall curve saved to {pr_curve_path}")


    logging.info(f"Merged overlapping hits: {len(hits_merged)} final candidate sites.")

    # Add nearest gene annotation to the main predictions file
    if args.gff:
        genes = []
        with open(args.gff) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                ftype = parts[2]
                if ftype != "gene" and ftype != "CDS":  # Look for both gene and CDS features
                    continue
                start, end, strand, attr = int(parts[3]), int(parts[4]), parts[6], parts[8]
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
                if gene_name:  # Only add if we found a gene name
                    genes.append((start, end, gene_name))

        # Add nearest gene to the merged hits for the main predictions file
        for hit in hits_merged:
            mid = (hit["start"] + hit["end"]) // 2
            hit["nearest_gene"] = min(genes, key=lambda g: min(abs(mid - g[0]), abs(mid - g[1])))[2] if genes else "NA"

    df_out = pd.DataFrame(hits_merged)
    out_path = os.path.join(outdir, f"{args.gene}_predictions.csv")
    df_out.to_csv(out_path, index=False)
    if supervised:
        logging.info(f"Predictions saved to {out_path} with gene annotation")
    else:
        logging.info(f"Predictions saved to {out_path} (UNSUPERVISED mode) with gene annotation")

    # ==============
    # VISUALIZATIONS 
    # ==============

    # 1. FEATURE IMPORTANCE PLOT (Only for supervised mode)
    if supervised:
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

    # 2. GENOME DISTRIBUTION PLOT
    if hits_merged:
        try:
            plt.figure(figsize=(14, 6))
            positions = [(hit['start'] + hit['end']) // 2 for hit in hits_merged]
            scores = [hit['score'] for hit in hits_merged]
            
            plt.scatter(positions, scores, alpha=0.6, s=30, c=scores, cmap='viridis')
            plt.colorbar(label='Prediction Score')
            plt.xlabel('Genomic Position')
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
            
            if supervised:
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

    # 4. PROMOTER CONTEXT PLOT
    if args.gff and hits_merged:
        try:
            # Extract TSS positions from promoters
            tss_positions = []
            for _, row in promoters.iterrows():
                if row["strand"] == "+":
                    tss_positions.append(row["prom_start"] + int(args.up))
                else:
                    tss_positions.append(row["prom_end"] - int(args.down))
            
            if tss_positions:
                plt.figure(figsize=(12, 6))
                
                # Calculate distance to nearest TSS for each hit
                distances = []
                for hit in hits_merged:
                    hit_mid = (hit['start'] + hit['end']) // 2
                    min_dist = min([abs(hit_mid - tss) for tss in tss_positions])
                    distances.append(min_dist)
                
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

# ---------------------------
# ARGPARSE CONFIGURATION
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid TFBS predictor: PWM + shape + RF")
    parser.add_argument('--fasta', required=True, help="Genome FASTA file path")
    parser.add_argument('--motif', required=True, help="TF motif file (JASPAR or MEME PFM)")
    parser.add_argument('--gene', required=True, help="TF name")
    parser.add_argument('--genome_accession', required=True, help="Genome accession ID in FASTA")
    parser.add_argument('--output', dest='outdir', default='results', help="Output directory")
    parser.add_argument('--n_trees', default=300, type=int, help="Number of trees in Random Forest")
    parser.add_argument('--neg_ratio', default=5, type=int, help="Ratio of negative to positive examples")
    parser.add_argument('--prob_cutoff', default=0.5, help="Probability cutoff for candidate site")
    parser.add_argument('--pad', default=10, help="Window padding around known sites")
    parser.add_argument('--seed', default=42, help="Random seed for reproducibility")
    parser.add_argument('--batch', default=10000, type=int, help="Batch size for genome scanning")
    parser.add_argument('--auto_cutoff', action='store_true', help="Automatically select probability cutoff using PR curve")
    parser.add_argument("--up", default=300, type=int, help="Upstream window size for promoter region")
    parser.add_argument("--down", default=50, type=int, help="Downstream window size for promoter region")
    parser.add_argument("--gff", required=True, help="Genome annotation in GFF3 format (used to define promoter regions)")
    parser.add_argument("--known", help="Known binding sequences (plain txt, one sequence per line)")
    parser.add_argument("--full_genome", action='store_true', help="Enable full-genome scanning instead of promoter-only")

    args = parser.parse_args()
    run(args)