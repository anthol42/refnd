# Ablation Study — HNSW → Leiden Pipeline

Standalone CLI that ablates HNSW parameters across three biological dataset modalities and evaluates community-based dataset splitting quality.

**ALWAYS USE UV RUN! NOT python3!!**

---

## Running an experiment

```bash
uv run python main.py \
  --dataset dbaasp \
  --ef-construction 64 \
  --ef-init 1 \
  [--extend-candidates] [--keep-pruned-connections] [--use-heuristic] \
  [--strict-ef] [--threshold-based-neighbourhood] \
  --leiden-objective modularity   # or cpm
  --label "optional free-text tag"
```

Output: `results/{name}.json`. The experiment name encodes all boolean flags:
`{dataset}_{ef_construction}_{ef_init}[-EC][-KPC][-UH][-SEF][-TBN][-CPM]`

---

## File layout

| File | Role |
|------|------|
| `main.py` | CLI entry point; orchestrates the full pipeline |
| `cache.py` | `CacheStore` — persists edges (`.edgestr`), embeddings (`.pth`), datasets (`.pkl`) under `.cache/` |
| `datasets.py` | Dataset registry (`DATASETS` dict) + download logic |
| `embeddings.py` | Foundation model embedding computation (ESM-C, ChemBERTa, DNABERT-2) |
| `metrics.py` | All metric helpers + null model + community-based split logic |
| `mlp.py` | Two-layer MLP (Linear→ReLU→Linear) with early stopping |

---

## Datasets

| Key | Modality | Kernel | Threshold | Encoder | Task |
|-----|----------|--------|-----------|---------|------|
| `dbaasp` | Peptide sequences | GlobalAligner + BLOSUM62 | 0.5 | ESM-C 300M | PCC (log MIC) |
| `ld50_zhu` | Morgan fingerprints | TanimotoBit | 0.4 | ChemBERTa | PCC (log LD50) |
| `prom_core_all` | DNA sequences | GlobalAligner + Identity | 0.8 | DNABERT-2 | MCC (binary) |

All kernels return **distance** (lower = more similar). Edges exist where distance ≤ threshold.

---

## Pipeline (per run)

1. **Load dataset** from cache or auto-download
2. **Compute embeddings** via frozen foundation model (cached per dataset)
3. **Gamma**: if `--leiden-objective cpm`, compute via `null_model()` (permutation null); else `gamma = 1.0`
4. **Exact edges** — O(n²), cached as `{dataset}_exact.edgestr`
5. **Build HNSW** (timed)
6. **For each graph type** (`edges` from `.edges()`, `layer0` from `.get_layer(0)`):
   - Community detection on HNSW graph (timed)
   - Compute all metrics (see below)
7. **Write JSON** to `results/{name}.json`

---

## Metrics (per graph type)

- `pct_edges_recovered` — |HNSW ∩ exact| / |exact|
- `missed_edge_weight_dist` — mean/std/median/p10/p25/p75/p90 of distances of missed edges
- `pct_missed_inter_community` — fraction of missed edges that cross communities in the exact graph
- `top3_components` — size + community count for the 3 largest connected components (HNSW graph)
- `split` — 10-repeat community-based train/test splits, each with a community-based val split for early stopping:
  - `no_postfilter` and `postfilter` variants
  - Reports `p_violations_mean/std[%]`, `mlp_score_mean/std`, `n_test_mean`, `n_train_mean`

### Null model (`null_model`)

Estimates P(distance ≤ threshold) under a random baseline by shuffling each sample's elements (amino acids / bits / nucleotides), sampling 10M pairs, and running the refnd kernel in parallel across CPU threads. Used as gamma for CPM Leiden.

### Community-based val split

The inner val split (for MLP early stopping) mirrors the outer test split: builds a sub-graph restricted to train nodes, re-runs `partition` on it with `test_ratio=0.15`. This avoids data leakage in the validation set.

---

## MLP

Architecture: `Linear(d, 2d) → ReLU → Linear(2d, out)`
- Regression (PCC): `out=1`, MSELoss, eval with Pearson r
- Classification (MCC): `out=n_classes`, CrossEntropyLoss, eval with Matthews CC
- Adam lr=0.003, max 300 epochs, early stopping patience=15, batch size=64

---

## Key implementation notes

- **DNABERT-2**: `transformers==4.57.6` (Biohub fork, pulled in by `esm`) rejects DNABERT-2's remote `BertConfig` in `AutoModel.register()`. Fix: monkey-patch `_BaseAutoModelClass.register` to set `model_class.config_class = config_class` and call original with `exist_ok=True`, restored in `finally`.
- **ESM-C pooling**: mask BOS/EOS/PAD tokens, mean-pool remaining positions.
- **ChemBERTa**: data is stored as `BitFingerprint`; SMILES strings are cached separately under `{dataset}_smiles` for the tokenizer.
- **LD50 fingerprints**: stored as `np.array(dtype=bool)` (BitFingerprint is not picklable); reconstructed via `BitFingerprint.from_np()` on load.
- **`requires-python = ">=3.13,<3.14"`**: upper bound avoids phantom resolver conflicts from `esm`'s Windows/Python 3.14 markers.
