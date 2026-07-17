"""Per-stage timing breakdown for the refnd HNSW->Leiden pipeline, across dataset sizes.

Post-filtering is disabled so `partition()` timing isn't polluted by its
violation-checking cost.

Usage:
    uv run python test_scaling_detailed_refnd.py
"""
import json
import time
from pathlib import Path

import numpy as np

from refnd.core import HNSWState, LeidenObjective, find_communities, partition

from cache import CacheStore
from datasets import DATASETS, load_dataset

SIZES   = [5_000, 25_000, 125_000, 625_000, 1_250_000]
SEED    = 42
TMP_DIR = Path(".cache/scaling_tmp")
RESULTS = Path("results/scaling_detailed_refnd.json")


def _subset_path(size: int) -> Path:
    return TMP_DIR / f"peptide_atlas_{size}.fasta"


def _write_fasta(path: Path, sequences: list[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def _prepare_subsets() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    needed = [s for s in SIZES if not _subset_path(s).exists()]
    if not needed:
        return
    print("Loading peptide_atlas dataset to generate missing subsets...")
    data, _ = load_dataset("peptide_atlas", CacheStore())
    print(f"  {len(data):,} sequences")
    rng = np.random.default_rng(SEED)
    for size in needed:
        if size > len(data):
            print(f"  [skip] size={size:,}: only {len(data):,} samples available")
            continue
        idx = rng.choice(len(data), size=size, replace=False)
        _write_fasta(_subset_path(size), [data[i] for i in idx])
        print(f"  Cached subset of {size:,} → {_subset_path(size)}")


def _read_fasta(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


def _time_one(size: int) -> dict | None:
    path = _subset_path(size)
    if not path.exists():
        print(f"  [skip] size={size:,}: subset not available")
        return None

    sequences = _read_fasta(path)
    cfg = DATASETS["peptide_atlas"]

    t0 = time.perf_counter()
    hnsw = HNSWState(cfg.modality, sequences, proximity_threshold=cfg.proximity_threshold, **cfg.kernel_params)
    hnsw.build(progress=True)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    es = hnsw.edges()
    t_edges = time.perf_counter() - t0
    n_edges = len(es)

    t0 = time.perf_counter()
    graph = es.graph(weighted=True, is_weight_distance=True)
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    coms = find_communities(graph, gamma=1.0, objective=LeidenObjective.Modularity)
    t_leiden = time.perf_counter() - t0

    t0 = time.perf_counter()
    partition(coms, graph, test_ratio=0.2, post_filtering=False)
    t_partition = time.perf_counter() - t0

    record = {
        "size": size,
        "n_edges": n_edges,
        "build_s": round(t_build, 3),
        "edges_s": round(t_edges, 3),
        "graph_s": round(t_graph, 3),
        "leiden_s": round(t_leiden, 3),
        "partition_s": round(t_partition, 3),
        "total_s": round(t_build + t_edges + t_graph + t_leiden + t_partition, 3),
    }
    print(f"  size={size:>10,}  n_edges={n_edges:,}  build={t_build:.2f}s  edges={t_edges:.2f}s  "
          f"graph={t_graph:.2f}s  leiden={t_leiden:.2f}s  partition={t_partition:.2f}s")
    return record


def main() -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    _prepare_subsets()
    records = []
    for size in SIZES:
        record = _time_one(size)
        if record is not None:
            records.append(record)
            RESULTS.write_text(json.dumps(records, indent=2))
    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()
