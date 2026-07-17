"""HNSW → Leiden → partition. Usage: uv run python hnsw_leiden_partition.py <fasta>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import DATASETS
from refnd.core import HNSWState

fasta = Path(sys.argv[1])
sequences = [line.strip() for line in fasta.read_text().splitlines()
             if line.strip() and not line.startswith(">")]

cfg   = DATASETS["peptide_atlas"]
hnsw  = HNSWState(cfg.modality, sequences, proximity_threshold=cfg.proximity_threshold, **cfg.kernel_params)
hnsw.build(progress=True)
