"""Scaling benchmark: split pipeline runtime vs. dataset size for peptide_atlas.

Usage:
    uv run python scaling_benchmark.py

Each method in METHODS is a standalone Python script that takes a single
positional argument (the input FASTA file). The orchestrator runs it under
/usr/bin/time to capture true peak RSS including Rust heap allocations.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from rich import print

from cache import CacheStore
from datasets import load_dataset

SIZES   = [5_000, 25_000, 125_000, 625_000, 1_250_000]
SEED    = 42
TIMEOUT = 86_400           # 1 day in seconds
RESULTS = Path("results/runtime.json")
TMP_DIR = Path(".cache/scaling_tmp")
METHODS_DIR = Path(__file__).parent / "runtime_scripts"

# ── Method registry: name → script path ───────────────────────────────────────

METHODS: dict[str, Path] = {
    # "refnd": METHODS_DIR / "split_refnd.py",
    # "hestia": METHODS_DIR / "split_hestia.py",
    "hnsw-only": METHODS_DIR / "split_hnsw_only.py",
}

# ── FASTA helpers ──────────────────────────────────────────────────────────────

def _write_fasta(path: Path, sequences: list[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


# ── Subset preparation ─────────────────────────────────────────────────────────

def _subset_path(size: int) -> Path:
    return TMP_DIR / f"peptide_atlas_{size}.fasta"


def _prepare_subsets(data: list[str]) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for size in SIZES:
        path = _subset_path(size)
        if path.exists():
            continue
        if size > len(data):
            print(f"  [yellow]Skipping size {size:,}: only {len(data):,} samples available[/]")
            continue
        idx = rng.choice(len(data), size=size, replace=False)
        _write_fasta(path, [data[i] for i in idx])
        print(f"  Cached subset of {size:,} → {path}")


# ── Results I/O ────────────────────────────────────────────────────────────────

def _load_results() -> list[dict]:
    if RESULTS.exists():
        with open(RESULTS) as f:
            return json.load(f)
    return []


def _save_results(records: list[dict]) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(records, f, indent=2)


# ── Subprocess runner ──────────────────────────────────────────────────────────

_IS_MACOS = platform.system() == "Darwin"
_TIME_FLAG = "-l" if _IS_MACOS else "-v"


def _parse_peak_rss_bytes(stderr: str) -> int | None:
    """Parse peak RSS from /usr/bin/time stderr (bytes on macOS, kbytes on Linux)."""
    if _IS_MACOS:
        m = re.search(r"(\d+)\s+maximum resident set size", stderr)
        return int(m.group(1)) if m else None
    else:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
        return int(m.group(1)) * 1024 if m else None


def _run_subprocess(method_name: str, script: Path, size: int, input_path: Path) -> dict:
    """Run script under /usr/bin/time in a fresh process; return timing/memory record."""
    cmd = ["/usr/bin/time", _TIME_FLAG, "uv", "run", str(script), str(input_path)]

    t0     = time.perf_counter()
    status = "ok"
    peak_b = None
    try:
        result  = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        elapsed = time.perf_counter() - t0
        peak_b  = _parse_peak_rss_bytes(result.stderr)
        if result.returncode != 0:
            status = f"error: rc={result.returncode}"
            if result.stderr:
                print(f"    [red]{result.stderr.strip()[:400]}[/]")
    except subprocess.TimeoutExpired:
        elapsed = TIMEOUT
        status  = "timeout"

    return {
        "method":      method_name,
        "size":        size,
        "runtime_s":   round(elapsed, 3),
        "peak_mem_mb": round(peak_b / 1024 ** 2, 2) if peak_b is not None else None,
        "status":      status,
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    cache = CacheStore()

    print("[bold][orange2]=== Peptide Atlas Scaling Benchmark ===[/][/]")
    print("Loading peptide_atlas dataset...")
    data, _ = load_dataset("peptide_atlas", cache)
    print(f"  {len(data):,} sequences")

    print("\nPreparing subsets...")
    _prepare_subsets(data)

    records = _load_results()

    for method_name, script in METHODS.items():
        print(f"\n[green]-- Method: {method_name} --[/]")
        for size in SIZES:
            input_path = _subset_path(size)
            if not input_path.exists():
                print(f"  [dim]Skipping size {size:,}: subset not available[/]")
                continue

            print(f"  size={size:>10,} ... ", end="", flush=True)
            record = _run_subprocess(method_name, script, size, input_path)
            records.append(record)
            _save_results(records)

            s     = record["status"]
            mem   = record["peak_mem_mb"]
            color = "green" if s == "ok" else "yellow" if s == "timeout" else "red"
            mem_s = f"{mem:.0f} MB" if mem is not None else "N/A"
            print(f"[{color}]{record['runtime_s']:.1f}s  {mem_s}  ({s})[/]")

            if s == "timeout":
                print(f"  [yellow]Timeout — skipping larger sizes for {method_name}[/]")
                break

    print(f"\n[bold]Results saved to {RESULTS}[/]")


if __name__ == "__main__":
    main()
