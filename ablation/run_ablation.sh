#!/usr/bin/env bash



# Do a scaling experiment with different subsets of PeptideAtlas
# Re-do ef-init with peptide atlas to see if same conclusions

# DBAASP
# Baseline
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-construction
uv run python main.py --dataset dbaasp        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-init
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# Turn off options
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm || true
# Turn on options
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm || true
uv run python main.py --dataset dbaasp        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm || true


# LD50
# Baseline
uv run python main.py --dataset ld50_zhu      --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-construction
uv run python main.py --dataset ld50_zhu        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-init
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# Turn off options
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm || true
# Turn on options
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm || true
uv run python main.py --dataset ld50_zhu        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm || true

# DNA
# Baseline
uv run python main.py --dataset prom_core_all --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-construction
uv run python main.py --dataset prom_core_all        --ef-construction 4 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 8 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 16 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 32 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 128 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 256 --ef-init 1 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# ef-init
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 2 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 4 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 8 --keep-pruned-connections --use-heuristic --leiden-objective cpm || true
# Turn off options
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --use-heuristic --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --leiden-objective cpm || true
# Turn on options
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --extend-candidates --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --strict-ef --leiden-objective cpm || true
uv run python main.py --dataset prom_core_all        --ef-construction 64 --ef-init 1 --keep-pruned-connections --use-heuristic --threshold-based-neighbourhood --leiden-objective cpm || true

