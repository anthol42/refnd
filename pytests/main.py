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