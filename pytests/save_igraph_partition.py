"""Runs igraph's reference Leiden once on the production edgestore dataset and
saves the resulting partition (membership vector) to disk as a `.npy` file,
next to the source dataset.

igraph's Leiden is expensive at this scale (building the Python-side
igraph.Graph over 100M+ edges alone takes minutes) and stochastic, so there's
no reason to pay that cost more than once: this script is prep, not the
benchmark itself. Later comparisons (refnd's achieved quality/timing vs
igraph's) load the saved membership and igraph's own `.quality` value instead
of re-running igraph.

Not a pytest test -- run manually.

Usage:
    uv run python save_igraph_partition.py <path.edgestr> [gamma] [beta] [n_iterations]
"""
import json
import sys
import time

import igraph as ig
import numpy as np

from refnd.core import EdgeStore

DEFAULT_PATH = "/mnt/documents/refnd_cache/threshold_belka/combined_train_test_layer0_0.2.edgestr"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.000008
    beta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    n_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    stem = path.rsplit(".", 1)[0]
    membership_path = f"{stem}.igraph_cpm_g{gamma}_b{beta}_i{n_iterations}.membership.npy"
    meta_path = f"{stem}.igraph_cpm_g{gamma}_b{beta}_i{n_iterations}.meta.json"

    print(f"Loading {path} ...")
    t = time.time()
    store = EdgeStore.load(path)
    edges = store.edges()
    print(f"  done in {time.time() - t:.2f}s ({store.node_count()} nodes, {len(edges)} edges)")

    edge_pairs = [(u, v) for u, v, _ in edges]
    weights = [w for _, _, w in edges]

    print("Building igraph.Graph ...")
    t = time.time()
    g = ig.Graph(n=store.node_count(), edges=edge_pairs)
    print(f"  done in {time.time() - t:.2f}s")

    print(f"Running igraph Leiden (CPM, resolution={gamma}, beta={beta}, n_iterations={n_iterations}) ...")
    t = time.time()
    result = g.community_leiden(
        objective_function="CPM", weights=weights, resolution=gamma, beta=beta, n_iterations=n_iterations
    )
    ig_time = time.time() - t
    n_communities = len(set(result.membership))
    print(f"  igraph: {ig_time:.3f}s, quality={result.quality:.4f}, communities={n_communities}")

    membership = np.asarray(result.membership, dtype=np.int32)
    np.save(membership_path, membership)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source": path,
                "gamma": gamma,
                "beta": beta,
                "n_iterations": n_iterations,
                "igraph_version": ig.__version__,
                "time_s": ig_time,
                "quality": result.quality,
                "n_communities": n_communities,
                "n_nodes": store.node_count(),
                "n_edges": len(edges),
            },
            f,
            indent=2,
        )
    print(f"Saved membership -> {membership_path}")
    print(f"Saved metadata   -> {meta_path}")


if __name__ == "__main__":
    main()
