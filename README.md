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
scans entire genomes in minutes and ranks every candidate site with a **Random Forest probability score**,
so sensitivity can be traded against precision with a single threshold.

| 🧬 Hybrid features | 🎯 Ranked probability scores | 🌍 Cross-genome | ⚡ Fast & reproducible |
|---|---|---|---|
| PWM log-odds + GC + Shannon entropy + **DNA shape** (minor groove width, propeller twist, roll, helical twist) from the published [DNAshape](https://doi.org/10.1093/nar/gkt437) pentamer table (Zhou et al., 2013) | Random Forest probabilities; operating threshold selected from **out-of-bag** predictions | A model trained on *E. coli* FNR sites (RegulonDB) transfers directly to *Salmonella* | Seeded end-to-end (`--seed 42` default), version-pinned dependencies, multi-core `joblib` parallelization |

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

- ✅ Recovers **75–96%** of known promoter-localized sites (AmrZ 75%, GlxR 76%, CodY 96%, E. coli FNR 91%)
- ✅ Site-level recovery broadly comparable to FIMO overall; FIMO recovers more sites at stringent
  matched prediction counts for three of four factors (significantly so for FNR)
- ✅ DNA-shape features improve ranking consistently across folds (largest gain for CodY)
- ✅ **Cross-genome transfer**: E. coli-trained FNR model → *Salmonella* Typhimurium LT2

## 🔬 Evaluation workflow (`evaluation/`)

The `evaluation/` directory contains the complete, deterministic (seed 42) evaluation behind the paper:

```
evaluation/
├── data/       curated datasets: cleaned CollecTF export, RegulonDB FNR sites,
│               LC–MS FNR regulon (Behera et al. 2020) gene mapping, reference genomes,
│               LT2 operon map, DNAshape pentamer query table
├── scripts/    the analysis pipeline
│   ├── cross_validation.py / eval_common.py   grouped 5-fold site-level CV (leakage-asserted)
│   ├── run_fimo_fair.py                       FIMO on identical promoter windows (same motif,
│   │                                          genome background, matched prediction counts)
│   ├── ablation.py / ablation_significance.py feature-ablation evaluation + fold-level statistics
│   ├── sitelevel_pr_evaluation.py             site-level precision–recall evaluation
│   ├── paired_site_recovery.py                paired per-site D-Sites-vs-FIMO recovery tests
│   ├── fnr_operon_matched_counts.py           FNR cross-genome transfer + operon-level tests
│   └── make_figures.py                        regenerates all manuscript figures
├── results/    all numerical outputs, grouped by analysis:
│   ├── paired/        paired site-recovery statistics (exact McNemar, Holm, bootstrap CIs)
│   ├── fnr_operon/    operon-level enrichment at matched counts (incl. no-shape control model)
│   ├── ablation/      feature-ablation statistics + FNR ablation arm
│   ├── sitelevel_pr/  site-level PR data and pooled scores
│   └── (plus the underlying CV/FIMO artifacts: oof_*.npz, fimo_scores_*.npz, cv_*.json)
├── tables/     summary tables (matched counts, operon enrichment, ablation)
└── logs/       execution logs with software versions, commands, and seeds
```

All analyses read the cached window universes and pooled out-of-fold scores, so every reported number
regenerates deterministically without retraining. See `evaluation/results/README.md` for commands.

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

**Pankaj** — Indian Institute of Technology Delhi · ft.pank@gmail.com
**Kanaka KK** — ICAR-Indian Institute of Agricultural Biotechnology · kkokay07@gmail.com
