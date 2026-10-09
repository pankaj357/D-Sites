#!/usr/bin/env python3
"""
extract_pqt.py — one-time extraction of the DNAshape pentamer query table
(PQT) from DNAshapeR's C++ source (TableCompiler.cpp, QueryTable array) into a
compact TSV shipped with D-Sites.

Column layout of the QueryTable entries (verified against
properties.cpp::load_data_from_vector, DNAshapeR v1.38.0):
  dv[0]=MGW  dv[6]=ProT  dv[15]=Roll1  dv[18]=Roll2  dv[21]=HelT1(twist1)
  dv[24]=HelT2(twist2)   (each stored as ave/sd/n triplets; we keep ave)
Roll1/HelT1 = value of the pentamer's first central bp-step; Roll2/HelT2 =
second central bp-step (see prediction.cpp::predict_step_parameters).

Outputs:
  Software/D-Sites/src/dsites/dnashape_pqt.tsv
  Original/src/dnashape_pqt.tsv   (identical copy)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/tmp/TableCompiler.cpp")
if not SRC.exists():
    sys.exit("TableCompiler.cpp not found at /tmp — re-download from "
             "https://raw.githubusercontent.com/TsuPeiChiu/DNAshapeR/master/src/TableCompiler.cpp")

text = SRC.read_text()
entries = re.findall(r'"([ATGCMQ]{5} [^"]+)"', text)
rows = []
seen = set()
for e in entries:
    f = e.split()
    assert len(f) == 91, f"entry has {len(f)} fields (expected 91): {f[:5]}..."
    pent = f[0]
    if "M" in pent or "Q" in pent:
        continue          # inosine-variant entries: unused for ACGT sequences
    v = [float(x) for x in f[1:]]
    rows.append((pent, v[0], v[6], v[15], v[18], v[21], v[24]))
    seen.add(pent)
assert len(entries) == 987, f"expected 987 QueryTable entries, got {len(entries)}"
assert len(rows) == 512, f"expected 512 ATGC-only entries, got {len(rows)}"

# coverage: every revcomp pair must have >=1 entry (D-Sites lookup falls back
# to the reverse complement, as DNAshapeR does via status -1)
_comp = str.maketrans("ACGT", "TGCA")
def rc(s):
    return s.translate(_comp)[::-1]
pairs = {}
for p in seen:
    pairs.setdefault(min(p, rc(p)), []).append(p)
assert len(pairs) == 512 and all(len(v) == 1 for v in pairs.values()), \
    "expected exactly one ATGC entry per reverse-complement pair"
n_both = 0
n_pairs = len(pairs)

HEADER = f"""# D-Sites DNA shape pentamer query table (PQT)
# Source: QueryTable array of DNAshapeR v1.38.0 (Bioconductor),
#   src/TableCompiler.cpp, Tsu-Pei Chiu & Federico Comoglio (Rohs Lab, USC).
# Method reference: Zhou T, Yang L, Lu Y, Dror I, Dantas Machado AC, Ghane T,
#   Di Felice R, Rohs R (2013) DNAshape: a method for the high-throughput
#   prediction of DNA structural features on a genomic scale.
#   Nucleic Acids Res 41:W56-W62. And: Li J et al. (2017) NAR 45:W281-W287.
# Column identities verified against DNAshapeR properties.cpp
#   (load_data_from_vector) and prediction.cpp (predict_groove_width /
#   predict_step_parameters).
# {len(rows)} pentamer entries: exactly ONE strand per reverse-complement
#   pair (the canonical member); the reverse complement's values are used
#   with the two step-value roles swapped (Roll1<->Roll2, HelT1<->HelT2),
#   exactly as DNAshapeR's status=-1 lookups do. The 475 inosine (M/Q)
#   entries of the source QueryTable are intentionally omitted (D-Sites
#   scans ACGT-only windows).
# Columns: pentamer, MGW (central bp), ProT (central bp),
#   Roll1/Roll2 (first/second central bp-step), HelT1/HelT2 (same).
# A D-Sites adoption note: replaces the previous hardcoded, untraceable
#   16-dinucleotide table; see Revision/results/shape_ab_test/ for the
#   protocol-identical A/B test justifying the swap.
pentamer\tMGW\tProT\tRoll1\tRoll2\tHelT1\tHelT2"""

out1 = ROOT / "Software" / "D-Sites" / "src" / "dsites" / "dnashape_pqt.tsv"
out2 = ROOT / "Original" / "src" / "dnashape_pqt.tsv"
body = "\n".join("\t".join([r[0]] + [f"{x:g}" for x in r[1:]]) for r in rows)
for out in (out1, out2):
    out.write_text(HEADER + "\n" + body + "\n")
    print(f"wrote {out} ({len(rows)} rows)")
print(f"coverage: {n_pairs} revcomp pairs, exactly one strand each (canonical)")
print("sanity: AAAAA =", [r for r in rows if r[0] == "AAAAA"][0])
