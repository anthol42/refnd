# Refnd
In datasets generated using a RGP (Relational Generative Process), such as datasets generated using evolution-like processes, the relational structure is important to consider for multiple tasks such as split without leakage, visualization, hypothesis generation / validation, etc. The relational structure of the dataset consists of knowing which pairs of elements are related (for example, which samples are evolutionary related). However, this structure is rarely known in advance, so we need to infer it.

Given a distance measurement, we can brute-force compute all pair distances to find related samples. Then, we can define a distance threshold under which samples are considered related. Linking those samples with an edge and the distance as weight yields a `thresholded-proximity graph`.

The problem with the brute force approach is that it has an $O(n^2)$ computational complexity, and does not scale well for large datasets. Fortunately, we defined a variant of the Hierarchical Navigable Small World (HNSW) algorithm to build this `thresholded proximity graph` in $O(nlog(n))$ instead.

Once the graph is obtained, operations on the dataset become easier, and more theoretically grounded. In fact, the distance in the graph between two samples can correlate with the likelihood of two samples being related if the distance measurement is well chosen. Hence, this helps visualize the data, split datasets without leakage, make discoveries, etc.

Furthermore, we can cluster the graph by finding communities or connected components. From these clusters, we can effectively **split the dataset into train and test set without leakage** with respect to the `proximity threshold` by splitting along clusters.

This library contains a toolkit of efficient functions and data structures to work with datasets generated from RGP. The core computations are implemented in Rust and multithreaded for maximum throughput! Everything is wrapped within an easy to use Python API. It currently supports:
- Protein/peptides sequences with Local and Global alignments
- Molecules with Real and Bit based Tanimoto similarities.
- More coming!

To give an idea of what the library contains, we have these functions:
- HNSW approximate proximity graph in $O(nlog(n))$
- Find nearest neighbors using an exact algorithm ($O(n)$) or approximate using HNSW ($O(log(n))$)
- Exact proximity graph in $O(n^2)$
- Leiden clustering to find communities in the graph.
- Find connected components within the graph.
- Partition a dataset along clusters to prevent data leakage.
- And more!

## Installation – Python
```shell
pip install refnd
```

Build from source (latest version, potentially unstable)
```shell
pip install "git+https://github.com/anthol42/refnd.git#subdirectory=py"
```

## Example
The following example shows how to split a protein dataset using 1 - global alignment as the distance function using the python API.
```python
from refnd import KernelVariant, HNSWState, find_communities, find_components, partition
from refnd.utils import read_fasta

# Load the dataset
dataset = read_fasta("datasets/proteins.fasta")
sequences = [seq for header, seq in dataset]

# Initiate the HNSW index
hnsw = HNSWState(KernelVariant.ProteinGlobal, sequences, proximity_threshold=0.3)

# Build it
hnsw.build()

# Get the proximity edges
edges = hnsw.edges()

# Load the graph
g = edges.graph()

# Get clusters
clusters = find_components(g) # Component based clustering – faster
clusters = find_communities(g) # Community based clustering - smaller clusters

# Partition into train and test
train_ids, test_ids = partition(clusters, g, post_filtering=True)
train = [dataset[i] for i in train_ids]
test = [dataset[i] for i in test_ids]
```

## Documentation
See the documentation references here: [https://anthol42.github.io/refnd/](https://anthol42.github.io/refnd/)

## Feedback
This project is currently in active development, and **your feedback is greatly appreciated**. If you find a bug, or would like a new feature, or give your thoughts on the API, please open an issue and we will be happy to help.
