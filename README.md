# D-Sites: Hybrid TFBS Predictor for Bacterial Genomes

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computational tool for predicting transcription factor binding sites (TFBS) in bacterial genomes using hybrid PWM, DNA shape features, and Random Forest classification.

## 📁 Project Structure
D-Sites/
├── src/ # Main source code
│ └── D-Sites.py # Primary prediction pipeline
├── data/ # Data files
│ └── known_sites/ # Curated known binding sites (collectf_export.tsv)
├── examples/ # Complete validation datasets
│ ├── AmrZ/ # AmrZ (Pseudomonas aeruginosa)
│ ├── GlxR/ # GlxR (Corynebacterium glutamicum)
│ ├── CodY/ # CodY (Bacillus anthracis)
│ └── Fnr/ # FNR (Salmonella Typhimurium)
├── scripts/ # Analysis & benchmarking suite
│ ├── fullbench.py # Comprehensive benchmarking
│ ├── comprehensive_validation.py # Validation analysis
│ ├── fimo_fnr.py # FNR-FIMO comparison
│ ├── generate_pr_curves.py # PR curve generation
│ ├── generate_enrichment_plot.py # Enrichment plots
│ ├── P_R_Bar_plot.py # Precision-Recall plots
│ ├── master_analysis.py # Master analysis
│ ├── fnr_fig.py # FNR visualization
│ └── FNR.py # FNR utilities
├── figures/ # Generated visualizations
├── requirements.txt # Python dependencies
└── LICENSE # MIT License

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/pankaj357/D-Sites.git
cd D-Sites
pip install -r requirements.txt
```

### Basic Prediction
```bash
python src/D-Sites.py --fasta examples/AmrZ/genome.fasta \
                     --gff examples/AmrZ/annotation.gff \
                     --motif examples/AmrZ/motif.meme \
                     --gene AmrZ \
                     --genome_accession NC_002516.2
```

### Run Benchmarking
```bash
# Comprehensive benchmarking
python scripts/fullbench.py

# FNR-specific analysis
python scripts/fimo_fnr.py

# Generate validation plots
python scripts/generate_pr_curves.py
```

## 📊 Available Scripts
- **src/D-Sites.py**: Main prediction pipeline  
- **scripts/fullbench.py**: Comprehensive performance evaluation  
- **scripts/comprehensive_validation.py**: Validation across all TFs  
- **scripts/fimo_fnr.py**: FNR-specific FIMO comparison  
- **scripts/generate_pr_curves.py**: Precision-Recall curve generation  
- **scripts/generate_enrichment_plot.py**: Promoter enrichment analysis  
- **scripts/master_analysis.py**: Master analysis script  

## 🧪 Validation Datasets
Complete validation data for four transcription factors:
- **AmrZ**: Pseudomonas aeruginosa PAO1  
- **GlxR**: Corynebacterium glutamicum R  
- **CodY**: Bacillus anthracis Sterne  
- **FNR**: Salmonella enterica Typhimurium  

## 📈 Performance
D-Sites demonstrates:
- Up to 9.3× higher recall than FIMO  
- 3-4× higher precision in top predictions  
- 3.02-3.42× enrichment in promoter regions  
- 68.1% validation success for FNR regulon  

## 📝 Citation
If you use D-Sites in your research, please cite:

Pankaj et al. (2025). *D-Sites: A computationally efficient tool for predicting protein binding sites in bacterial genomes*. Journal Name, Volume, Pages.

## 📄 License
MIT License - see LICENSE for details.

## 💬 Contact
For questions and support, please open an issue on GitHub or contact **ft.pank@gmail.com**.
