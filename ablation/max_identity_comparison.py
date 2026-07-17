"""Compare max-identity-to-nearest-train-neighbor between an mmseqs2 cluster split
and a refnd community split (no post-filtering), on the dbaasp dataset.

For each test sample, "max-identity" is 1 - (distance to its nearest train
neighbor), i.e. the GlobalAligner/BLOSUM62 identity score. A leakage-free split
should push this distribution toward low identities; mmseqs2 (min-seq-id=0.5)
and refnd (proximity_threshold=0.5) are both clustering at the same identity
threshold, so their resulting distributions are directly comparable.

Usage:
    uv run python max_identity_comparison.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cache import CacheStore
from datasets import DATASETS, load_dataset
from metrics import null_model
from refnd.core import EdgeStore, HNSWState, LeidenObjective, find_communities, partition

DATASET    = "dbaasp"
MIN_SEQ_ID = 0.5
TEST_RATIO = 0.2
SEED       = 42
WORKDIR    = Path(".cache/mmseqs_max_identity")
RESULTS    = Path("results/max_identity_comparison.json")
FIGURE     = Path("results/max_identity_comparison.png")


# ── mmseqs2 cluster split ───────────────────────────────────────────────────

def _write_fasta(path: Path, sequences: list[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def _run_mmseqs_cluster(fasta_path: Path, workdir: Path) -> list[int]:
    """Run mmseqs easy-cluster and return a cluster ID per sequence (by index)."""
    cluster_prefix = workdir / "cluster"
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"mmseqs easy-cluster {fasta_path} {cluster_prefix} {tmp_dir} "
        f"--min-seq-id {MIN_SEQ_ID} "
        f"--alignment-mode 3 "
        f"--gap-open 11 "
        f"--gap-extend 1 "
        f"--cov-mode 1 "
        f"--cluster-mode 1 "
        f"--comp-bias-corr 0 "
        f"-s 7.5"
    )
    print(f"  Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    tsv_path = Path(f"{cluster_prefix}_cluster.tsv")
    rep_to_cluster: dict[str, int] = {}
    seq_to_cluster: dict[str, int] = {}
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            if rep not in rep_to_cluster:
                rep_to_cluster[rep] = len(rep_to_cluster)
            seq_to_cluster[member] = rep_to_cluster[rep]

    n = len(seq_to_cluster)
    clusters = [seq_to_cluster[f"seq_{i}"] for i in range(n)]
    return clusters


def mmseqs_split(sequences: list[str]) -> tuple[list[int], list[int]]:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    fasta_path = WORKDIR / "input.fasta"
    _write_fasta(fasta_path, sequences)

    clusters = _run_mmseqs_cluster(fasta_path, WORKDIR)
    print(f"  mmseqs2: {len(set(clusters))} clusters from {len(sequences)} sequences")

    empty_graph = EdgeStore(len(sequences), []).graph()
    train_idx, test_idx = partition(clusters, empty_graph, test_ratio=TEST_RATIO,
                                     seed=SEED, post_filtering=False)
    return list(train_idx), list(test_idx)


# ── refnd community split ───────────────────────────────────────────────────

def refnd_split(sequences: list[str]) -> tuple[list[int], list[int]]:
    cfg = DATASETS[DATASET]

    print("  Computing null model for CPM gamma...")
    gamma = null_model(sequences, cfg)
    print(f"  gamma (null model) = {gamma:.6f}")

    hnsw = HNSWState(cfg.modality, sequences, proximity_threshold=cfg.proximity_threshold,
                      **cfg.kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(weighted=True, is_weight_distance=True)

    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    print(f"  refnd: {len(set(communities))} communities from {len(sequences)} sequences")

    train_idx, test_idx = partition(communities, graph, test_ratio=TEST_RATIO,
                                     seed=SEED, post_filtering=False)
    return list(train_idx), list(test_idx)


# ── max-identity to nearest train neighbor ──────────────────────────────────

def max_identity_to_train(sequences: list[str], train_idx: list[int], test_idx: list[int]) -> np.ndarray:
    cfg = DATASETS[DATASET]
    train_data = [sequences[i] for i in train_idx]
    test_data  = [sequences[i] for i in test_idx]

    train_hnsw = HNSWState(cfg.modality, train_data, proximity_threshold=cfg.proximity_threshold,
                            **cfg.kernel_params)
    train_hnsw.build(progress=True)
    hits = train_hnsw.search(test_data, k=1, ef=64, threads=0, progress=True)
    distances = np.array([h[0][1] if h else np.nan for h in hits], dtype=float)
    return 1.0 - distances


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(identities: dict[str, np.ndarray]) -> None:
    colors = {"mmseqs2": "#1f77b4", "refnd": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 1, 41)
    for name, vals in identities.items():
        ax.hist(vals, bins=bins, alpha=0.5, color=colors[name], label=name, density=True)
        median = np.median(vals)
        p90 = np.percentile(vals, 90)
        ax.axvline(median, color=colors[name], linestyle="--", linewidth=1.5,
                   label=f"{name} median={median:.3f}")
        ax.axvline(p90, color=colors[name], linestyle=":", linewidth=1.5,
                   label=f"{name} p90={p90:.3f}")

    ax.set_xlabel("Max identity to nearest train neighbor")
    ax.set_ylabel("Density")
    ax.set_title(f"{DATASET}: mmseqs2 (min-seq-id={MIN_SEQ_ID}) vs refnd (threshold={MIN_SEQ_ID}) split")
    ax.legend(fontsize=8)
    plt.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    plt.savefig(FIGURE, dpi=150)
    plt.show()


def main() -> None:
    cache = CacheStore()
    print("Loading dataset...")
    sequences, _ = load_dataset(DATASET, cache)
    print(f"  {len(sequences)} sequences")

    print("\n[mmseqs2 split]")
    mmseqs_train, mmseqs_test = mmseqs_split(sequences)
    print(f"  train={len(mmseqs_train)}  test={len(mmseqs_test)}")

    print("\n[refnd split]")
    refnd_train, refnd_test = refnd_split(sequences)
    print(f"  train={len(refnd_train)}  test={len(refnd_test)}")

    print("\nComputing max-identity to nearest train neighbor...")
    identities = {
        "mmseqs2": max_identity_to_train(sequences, mmseqs_train, mmseqs_test),
        "refnd":   max_identity_to_train(sequences, refnd_train, refnd_test),
    }

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        name: {
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "values": vals.tolist(),
        }
        for name, vals in identities.items()
    }, indent=2))
    print(f"Results saved to {RESULTS}")

    plot_comparison(identities)
    print(f"Figure saved to {FIGURE}")


if __name__ == "__main__":
    main()
