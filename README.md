<div align="center">

# 🧬 D-Sites

**A hybrid machine-learning framework for prediction of transcription factor binding sites in bacterial genomes**

[![Version](https://img.shields.io/badge/version-1.3.0-blue.svg)](https://github.com/pankaj357/D-Sites)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](Software/D-Sites/LICENSE)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey.svg)]()

`pip install dsites` · `dsites --help`

</div>

---

## 🌟 Why D-Sites?

Identifying transcription factor binding sites (TFBS) is essential for mapping bacterial regulatory
networks — but experimental methods (EMSA, DNase footprinting, ChIP) are slow and costly. **D-Sites**
scans entire genomes in minutes and ranks every candidate site with a **calibrated probability score**,
so you can trade sensitivity against precision with a single threshold.

| 🧬 Hybrid features | 🎯 Calibrated scores | 🌍 Cross-genome | ⚡ Fast & reproducible |
|---|---|---|---|
| PWM log-odds + GC + Shannon entropy + **DNA shape** (minor groove width, propeller twist, roll, helical twist) from the published [DNAshape](https://doi.org/10.1093/nar/gkt437) pentamer table (Zhou et al., 2013) | Random Forest probabilities; operating threshold selected from **out-of-bag** predictions — never tuned on test data | A model trained on *E. coli* FNR sites (RegulonDB) transfers directly to *Salmonella* | Seeded end-to-end (`--seed 42` default), version-pinned dependencies, multi-core `joblib` parallelization |

## 🚀 Quick start

```bash
pip install dsites

dsites --fasta genome.fna \
       --gff annotation.gff3 \
       --motif TF.jaspar \
       --gene TF \
       --genome_accession NC_XXXXXXX \
       --output results_TF
```

Outputs: merged predictions with gene annotation (`{gene}_predictions.csv`), the full window-level set,
the top-5% high-confidence set, and diagnostic plots (feature importance, score distribution,
genome-wide map, distance-to-TSS).

## 📊 Held-out performance

Every number below comes from **grouped 5-fold site-level cross-validation**, with FIMO compared on
*identical promoter windows at matched prediction counts* — the full workflow, scripts, and raw results
are in [`evaluation/`](evaluation/results/README.md):

- ✅ Recovers **75–96%** of known promoter-localized sites (AmrZ 75%, GlxR 76%, CodY 96%)
- ✅ Ranking quality comparable to FIMO overall; **beats FIMO for CodY** (precision 0.154 vs 0.118 at N = 500)
- ✅ DNA-shape features improve ranking for **every** factor tested (ablation; ~2× PR-AUC gain for CodY)
- ✅ **Cross-genome transfer**: E. coli-trained FNR model → *Salmonella* Typhimurium LT2

## 🗂️ Repository layout

```
Software/D-Sites/   D-Sites source code (v1.3.0)
                    └── src/dsites/predictor.py · dnashape_pqt.tsv · pinned requirements
evaluation/         Complete held-out evaluation behind the paper
                    ├── data/      cleaned datasets (CollecTF, RegulonDB, LC–MS regulon, genomes)
                    ├── scripts/   cross-validation, fair-FIMO, ablation, cross-genome pipeline
                    └── results/   every JSON/CSV — all numbers regenerate deterministically
```

## ⚙️ How it works

```
FASTA + GFF3 + motif (JASPAR/MEME)
        │
        ▼  promoter windows (−300/+50 per CDS, ALL contigs incl. plasmids)
PWM log-odds ──► 11-feature vectors ◄── DNAshape pentamer features (MGW·ProT·Roll·HelT)
        │
        ▼  Random Forest (seeded, OOB-calibrated threshold)
ranked binding-site probabilities per window  ──►  CSVs + plots
```

## 📝 Key design points (v1.3.0)

- **PWM**: count-based pseudocount `p = (c + α)/(N + 4α)`, α = 0.5; natural-log log-odds.
- **Negatives**: sampled from the *same promoter windows* that are scanned; `--neg_ratio` honored exactly; `--seed` for full determinism.
- **Scanning**: sliding 1-bp windows, both strands, all contigs; optional `--full_genome` mode.
- **Novelty**: unlike PWM-only scanners, the learned ranking integrates sequence *and* structural readout — most valuable for factors whose binding depends on DNA shape.

## 📚 Citation

> Pankaj et al. *D-Sites: A hybrid machine-learning framework for prediction of transcription factor binding sites in bacterial genomes.* **Evolutionary Bioinformatics** (under review).

## 📄 License

[MIT](Software/D-Sites/LICENSE) — free for academic and commercial use.

## 💬 Contact

**Pankaj** — Indian Agricultural Research Institute (IARI) · ft.pank@gmail.com
**Kanaka KK** — ICAR-Indian Institute of Agricultural Biotechnology · kkokay07@gmail.com
