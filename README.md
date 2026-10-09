# D-Sites

**D-Sites: a hybrid machine-learning framework for prediction of transcription factor binding sites in bacterial genomes**

D-Sites integrates PWM log-odds scoring, sequence features (GC content, Shannon entropy), and DNA shape
features from the published [DNAshape](https://doi.org/10.1093/nar/gkt437) pentamer query table
(Zhou et al., 2013; values as distributed in DNAshapeR v1.38.0) to train a Random Forest classifier on
curated TFBS and genome-aware negative samples, producing calibrated per-window binding probabilities.
The tool scans promoter windows on **all contigs** of a bacterial assembly (including plasmids), is fully
seeded for reproducibility, and parallelizes feature extraction/prediction across CPU cores.

- Install: `pip install dsites` (v1.3.0) · CLI: `dsites --help`
- Publication: *Evolutionary Bioinformatics*. The complete held-out evaluation workflow supporting the paper is in [`evaluation/`](evaluation/results/README.md).

## Repository layout

| Directory | Contents |
|---|---|
| `Software/D-Sites/` | The D-Sites source code (v1.3.0): `src/dsites/predictor.py` is the pipeline; `src/dsites/dnashape_pqt.tsv` is the published DNAshape pentamer query table. The installable package is distributed via **PyPI** (`pip install dsites`). |
| `evaluation/` | The held-out evaluation behind the paper: cleaned datasets (`data/`), analysis scripts (`scripts/`), and all result files (`results/`) — every number in the paper is regenerable from these. See `evaluation/results/README.md` for reproduction commands. |

## Quick start

```bash
pip install dsites                  # v1.3.0 from PyPI
dsites --fasta genome.fna --gff annotation.gff3 --motif TF.jaspar \
       --gene TF --genome_accession NC_XXXXXXX --output results_TF
```

## Key design points (v1.3.0)

- PWM pseudocount: standard count-based correction `p = (c + α)/(N + 4α)`, α = 0.5; natural-log log-odds.
- Negatives sampled from the same promoter windows that are scanned; `--neg_ratio` honored exactly; `--seed` (default 42).
- Operating threshold selected from out-of-bag predictions (OOB-F1); top-5% file retained as a fixed-proportion set.
- Multi-contig scanning; optional `--full_genome` mode.
- Version-pinned dependencies (`requirements.txt`).

## License
MIT (see `Software/D-Sites/LICENSE`).
