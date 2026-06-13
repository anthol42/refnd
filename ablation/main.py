"""Ablation study CLI: HNSW → Leiden pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from rich import print
sys.path.insert(0, str(Path(__file__).parent))

from cache import CacheStore
from datasets import DATASETS, load_dataset
from embeddings import compute_embeddings
from metrics import compute_graph_metrics, layer0_to_edge_store, null_model

from refnd.core import (
    HNSWState,
    LeidenObjective,
    exact_edges,
    find_communities,
)


# ── Experiment naming ──────────────────────────────────────────────────────────

def make_name(args: argparse.Namespace) -> str:
    base = f"{args.dataset}_{args.ef_construction}_{args.ef_init}"
    flags = []
    if args.extend_candidates:               flags.append("EC")
    if args.keep_pruned_connections:         flags.append("KPC")
    if args.use_heuristic:                   flags.append("UH")
    if args.strict_ef:                       flags.append("SEF")
    if args.threshold_based_neighbourhood:   flags.append("TBN")
    if args.leiden_objective == "cpm":       flags.append("CPM")
    if flags:
        base += "-" + "-".join(flags)
    return base


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNSW ablation study")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))

    # HNSW parameters
    p.add_argument("--ef-construction",               type=int,   default=64)
    p.add_argument("--ef-init",                       type=int,   default=1)
    p.add_argument("--extend-candidates",             action="store_true")
    p.add_argument("--keep-pruned-connections",       action="store_true")
    p.add_argument("--use-heuristic",                 action="store_true")
    p.add_argument("--strict-ef",                     action="store_true")
    p.add_argument("--threshold-based-neighbourhood", action="store_true")

    # Leiden parameters
    p.add_argument("--leiden-objective", choices=["modularity", "cpm"], default="modularity")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    cfg   = DATASETS[args.dataset]
    cache = CacheStore()
    name  = make_name(args)

    print(f"[bold][orange2]=== Experiment: {name} ===[/][/]")

    # 1. Dataset + embeddings
    print("Loading dataset...")
    data, labels = load_dataset(args.dataset, cache)
    print(f"  {len(data)} samples")

    print("Loading/computing embeddings...")
    embs = compute_embeddings(args.dataset, data, cfg, cache)

    # Gamma: null model for CPM, fixed 1.0 for Modularity
    leiden_obj = (
        LeidenObjective.CPM if args.leiden_objective == "cpm"
        else LeidenObjective.Modularity
    )
    if args.leiden_objective == "cpm":
        print("Computing null model for CPM gamma...")
        gamma = null_model(data, cfg)
        print(f"  gamma (null model) = {gamma:.6f}")
    else:
        gamma = 1.0

    # 2. Exact edges (cached per dataset)
    exact_cache_key = f"{args.dataset}_exact"
    exact_es = cache.get_edges(exact_cache_key)
    if exact_es is None:
        print("Computing exact edges (O(n²), cached after this)...")
        exact_es = exact_edges(
            cfg.modality, data,
            proximity_threshold=cfg.proximity_threshold,
            **cfg.kernel_params,
        )
        cache.store_edges(exact_cache_key, exact_es)
    exact_graph = exact_es.graph(weighted=True, is_weight_distance=True)

    # 3. Build HNSW (timed)
    print("Building HNSW...")
    hnsw = HNSWState(
        cfg.modality, data,
        proximity_threshold=cfg.proximity_threshold,
        ef_construction=args.ef_construction,
        ef_init=args.ef_init,
        extend_candidates=args.extend_candidates,
        keep_pruned_connections=args.keep_pruned_connections,
        use_heuristic=args.use_heuristic,
        strict_ef=args.strict_ef,
        threshold_based_neighbourhood=args.threshold_based_neighbourhood,
        **cfg.kernel_params,
    )
    t0 = time.perf_counter()
    hnsw.build(progress=True)
    hnsw_build_time = time.perf_counter() - t0
    print(f"  Built in {hnsw_build_time:.2f}s")

    # 4. Per graph type
    graph_results: dict[str, dict] = {}

    for graph_type in ("edges", "layer0"):
        print(f"\n[green]-- Graph type: {graph_type} --[/]")

        if graph_type == "edges":
            graph_weighted     = True
            graph_is_distance  = True
            hnsw_es    = hnsw.edges()
            hnsw_graph = hnsw_es.graph(weighted=graph_weighted, is_weight_distance=graph_is_distance)
        else:
            graph_weighted     = False
            graph_is_distance  = False
            adj        = hnsw.get_layer(0)
            hnsw_es    = layer0_to_edge_store(adj, len(data))
            hnsw_graph = hnsw_es.graph(weighted=graph_weighted, is_weight_distance=graph_is_distance)

        # Community detection on HNSW graph (timed)
        t0 = time.perf_counter()
        hnsw_communities = find_communities(hnsw_graph, gamma=gamma, objective=leiden_obj)
        cd_time = time.perf_counter() - t0
        print(f"  [dim]{len(set(hnsw_communities))} communities in {cd_time:.2f}s[/]")

        graph_results[graph_type] = compute_graph_metrics(
            hnsw_es=hnsw_es,
            exact_es=exact_es,
            hnsw_communities=hnsw_communities,
            hnsw_graph=hnsw_graph,
            exact_graph=exact_graph,
            graph_weighted=graph_weighted,
            graph_is_distance=graph_is_distance,
            hnsw_cd_time=cd_time,
            hnsw_build_time=hnsw_build_time,
            data=data,
            labels=labels,
            embs=embs,
            variant=cfg.modality,
            proximity_threshold=cfg.proximity_threshold,
            metric=cfg.metric,
            kernel_params=cfg.kernel_params,
            com_objective=leiden_obj,
            gamma=gamma,
        )

    # 5. Write JSON
    output = {
        "name": name,
        "label": args.label,
        "input": {
            "dataset":                       args.dataset,
            "ef_construction":               args.ef_construction,
            "ef_init":                       args.ef_init,
            "extend_candidates":             args.extend_candidates,
            "keep_pruned_connections":       args.keep_pruned_connections,
            "use_heuristic":                 args.use_heuristic,
            "strict_ef":                     args.strict_ef,
            "threshold_based_neighbourhood": args.threshold_based_neighbourhood,
            "leiden_objective":              args.leiden_objective,
            "gamma":                         gamma,
        },
        "results": graph_results,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
