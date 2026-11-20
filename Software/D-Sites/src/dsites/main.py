# dsites/main.py
import argparse
import logging
from .predictor import run  # our main pipeline is in predictor.py
from . import __version__

BANNER = (
    f"D-Sites v{__version__} - hybrid TFBS prediction (PWM + DNA shape + Random Forest)\n"
    "Authors: Pankaj (IIT Delhi; contact: ft.pank@gmail.com), "
    "Kanaka KK (ICAR-Indian Institute of Agricultural Biotechnology, Ranchi; contact: kkokay07@gmail.com)\n"
    "GitHub: https://github.com/pankaj357/D-Sites"
)

def cli():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="D-Sites: Hybrid TFBS predictor (PWM + DNA shape + RF)"
    )
    # Mirror the arguments your script expects:
    parser.add_argument('--fasta', required=True, help="Genome FASTA file path")
    parser.add_argument('--motif', required=True, help="TF motif file (JASPAR or MEME PFM)")
    parser.add_argument('--gene', required=True, help="TF name")
    parser.add_argument('--genome_accession', required=True,
                        help="Genome accession ID used to look up known sites in the bundled "
                             "CollecTF dataset (scanning covers ALL contigs in the FASTA)")
    parser.add_argument('--output', '--outdir', dest='outdir', default='results', help="Output directory")
    parser.add_argument('--n_trees', default=300, type=int, help="Number of trees in Random Forest")
    parser.add_argument('--neg_ratio', default=5, type=int,
                        help="Ratio of negative to positive examples (honored exactly)")
    parser.add_argument('--min_negatives', default=0, type=int,
                        help="Optional minimum number of negative examples (default 0 = no minimum)")
    parser.add_argument('--pseudocount', default=0.5, type=float,
                        help="Count-based PWM pseudocount pc: p = (count + pc) / (N + 4*pc)")
    parser.add_argument('--prob_cutoff', default=0.5, type=float, help="Probability cutoff for candidate site")
    parser.add_argument('--pad', default=10, type=int, help="Window padding around known sites")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--batch', default=10000, type=int, help="Batch size for genome scanning")
    parser.add_argument('--auto_cutoff', action='store_true', help="Automatically select probability cutoff using PR curve")
    parser.add_argument("--up", default=300, type=int, help="Upstream window size for promoter region")
    parser.add_argument("--down", default=50, type=int, help="Downstream window size for promoter region")
    parser.add_argument("--gff", help="Genome annotation in GFF3 format (required for promoter-based "
                                      "scanning; optional with --full_genome, where it is still used "
                                      "for nearest-gene annotation)")
    parser.add_argument("--known", help="Known binding sequences (plain txt, one sequence per line)")
    parser.add_argument("--full_genome", action='store_true',
                        help="Slide windows across the entire length of every contig instead of "
                             "promoter windows only")

    args = parser.parse_args()
    print(BANNER)
    logging.info(BANNER.replace("\n", " | "))
    run(args)

if __name__ == "__main__":
    cli()
