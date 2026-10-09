# D-Sites: Hybrid TFBS Predictor for Bacterial Genomes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computational tool for predicting transcription factor binding sites (TFBS) in bacterial genomes using hybrid PWM, DNA shape features, and Random Forest classification.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/pankaj357/D-Sites.git
cd D-Sites
pip install .
```
This installs the `dsites` command-line tool and all required dependencies
(see `requirements.txt` for the exact pinned versions used for the published
benchmarks).

### Basic Prediction
#### Minimal Command
```bash
dsites --fasta <genome.fasta> \
       --gff <annotation.gff> \
       --motif <motif_file> \
       --gene <TF_name> \
       --genome_accession <accession_id>
```
#### Complete Example
```bash
dsites \
    --fasta <path_to_genome.fasta> \
    --gff <path_to_annotation.gff> \
    --motif <path_to_motif_file> \
    --gene <TF_NAME> \
    --genome_accession <GENOME_ACCESSION> \
    --output results \
    --n_trees 300 \
    --neg_ratio 5 \
    --prob_cutoff 0.5 \
    --pseudocount 0.5 \
    --pad 10 \
    --seed 42 \
    --batch 10000 \
    --up 300 \
    --down 50 \
    --auto_cutoff
```
If the TF/genome pair is not present in the bundled CollecTF export, provide
known binding sequences explicitly (plain text, one sequence per line):
```bash
dsites --fasta genome.fna --gff genomic.gff --motif motif.txt \
       --gene FNR --genome_accession NC_003197.2 \
       --known known_binding_sites.txt --output results
```

### Command Breakdown
#### Required Arguments
```bash
--fasta: Genome FASTA file path

--gff: Genome annotation file (GFF3 format, required for promoter-based
       scanning; optional with --full_genome)

--motif: TF motif file (JASPAR or MEME format)

--gene: Transcription factor name

--genome_accession: Genome accession ID, used to look up known sites in the
       bundled CollecTF dataset (scanning always covers ALL contigs in the FASTA)
```
#### Optional Arguments with Defaults
``` bash
--output results: Output directory (alias: --outdir)

--n_trees 300: Number of Random Forest trees

--neg_ratio 5: Negative:Positive ratio (honored exactly)

--min_negatives 0: Optional minimum number of negative examples

--pseudocount 0.5: Count-based PWM pseudocount, p = (count + pc) / (N + 4*pc)

--prob_cutoff 0.5: Probability cutoff

--pad 10: Window padding around known sites

--seed 42: Random seed (runs are fully deterministic for a fixed seed)

--batch 10000: Batch size for parallel scanning

--up 300: Upstream promoter size

--down 50: Downstream promoter size

--full_genome: Slide windows across entire contigs instead of promoter windows

--auto_cutoff: Automatically select probability cutoff via precision-recall curve
```

## 📈 Performance
D-Sites demonstrates:
- 3-4× higher precision in top predictions  
- 3.02-3.42× enrichment in promoter regions  

## 📝 Citation
If you use D-Sites in your research, please cite:

Pankaj et al. (2025). *D-Sites: A hybrid machine-learning framework for prediction of transcription factor binding sites in bacterial genomes*. Information Sciences (Under Review),2024.

## 📄 License
MIT License - see LICENSE for details.

## 💬 Contact
For questions and support, contact:
- Pankaj: ft.pank@gmail.com
- Kanaka KK: kkokay07@gmail.com
