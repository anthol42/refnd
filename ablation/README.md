

The HNSW parameters we will toggle are:
  - ef_construction
  - ef_init
  - extend_candidates
  - keep_pruned_connections
  - use_heuristic
  - strict_ef

  - Leiden objective: Modularity / CPM
  - Gamma

  We will extract two types of graph from the built HNSW: from `.edges()` and `.get_layer(0)`.

  The results will evaluate:
  - Runtime (Time to build the HNSW graph, and find communities, two values not aggregated)
  - The percentage of edges recovered (From the exact computation)
  - The distribution of weight of missed edges. (Report the mean, std, median, 25, 75, 10, 90 percentiles)
  - Make a community detection on the components of the exact proximity graph, and observe the percentage of missing
  edge that are infact inter-community edges. The community detection will default to Modularity, but can be changed
  to CPM if we know the gamma (null model probability)
  - Size of the three largest component, and their number of communities within them (Again, with Modularity by
  default, or CPM if we know gamma)
  - Split the dataset community-wise, and observe the maximal similarity to the nearest train neighbor for each test
  sample, and count the number that violates the threshold. Split with and without post-filtering, and report both
  values
  - Train a MLP model (2 layers, relu in-between and 2x neurons in the hidden layer) on train set, on top of the
  embeddings of a foundational model (Will be cached), then evaluate on the test set, and report test performances. Do
  this with both, post-filtering and no post-filtering


Protein et peptides:
- Encoder: ESM-C 300M
- Dataset: DBAASP from the QMAP-benchmark package
- Regression: PCC

Small molecules
- Encoder: ChemBERTa-zinc-base-v1
- Dataset: LD50 (Zhu)
- Regression: PCC

ADN
- Encoder: DNA-BERT2
- Dataset: prom_core_all in GUE dataset.
- Classification: MCC

Protein et peptides:
- Encoder: ESM-C 300M
- Dataset: DBAASP from the QMAP-benchmark package
- Regression: PCC

Small molecules
- Encoder: ChemBERTa-zinc-base-v1
- Dataset: LD50 (Zhu)
- Regression: PCC

ADN
- Encoder: DNA-BERT2
- Dataset: prom_core_all in GUE dataset.
- Classification: MCC


Notes:
- Download LD50 from this link: https://huggingface.co/datasets/scikit-fingerprints/TDC_ld50_zhu/resolve/main/tdc_ld50_zhu.csv
- Download prom_core_all from this link: https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/prom_core_all/train.csv


uv run python main.py --dataset prom_core_all --ef-construction 64 --ef-init 1 --use-heuristic 
--keep-pruned-connections --leiden-objective cpm