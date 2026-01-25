# Data files
base_filepath=/path/to/sift1b/bigann_100m_base.ubin
query_filepath=/path/to/sift1b/bigann_query.ubin
ground_truth_filepath=/path/to/sift1b/gnd/idx_1000M.ibin

# Data formats / types
dataset_fmt=u8bin                 # bvecs/ivecs/fbin/ibin
vector_type=uint8                 # uint8/int/float
queryset_fmt=u8bin                # bvecs/ivecs/fbin/ibin
truthset_fmt=ibin                 # ivecs/ibin

# Basic dataset properties
d=128
nb=100000000
nq=10000
ratio=10                            # sample ratio (100/ratio)% data used to build the index

# Partitions and paths
partitions=10
build_disk_store_path=/path/to/build/disk/store
build_centroid_index_path=/path/to/build/centroid/index
search_disk_store_path=/path/to/search/disk/store
search_centroid_index_path=/path/to/search/centroid/index

# Build phase parameters
nlist=200000                        # IVF cluster count
m=16                                # PQ sub-quantizers
nbits=8                             # PQ bit width
replicas=2                          # Replicas for building
shrink_replicas=1.15                # Shrink replicas by removing some points
build_estimate_factor=1.2           # Estimate factor for building (1 + epsilon)
build_prune_factor=1.9              # Prune factor for building
build_metric_type=L2                # IP/L2
build_memory_graph_efb=75           # HNSW in-memory graph efConstruction
build_memory_graph_efs=150          # HNSW in-memory graph efSearch
build_memory_graph_M=16             # HNSW graph degree (M) for building
build_threads=128                   # Threads for building

# Search phase parameters
k=100                               # candidates to retrieve
k_per_partition=40                  # retrieve k per partition candidates from each partition
search_top=2                        # Conservative pruning (t)
search_nprobes=200                  # Bucket number to search (s)
search_estimate_factors_partial=1.04
search_threads=16
cache_vectors=0
query_for_warm_up=0

# Recommended defaults
search_estimate_factor=1.05         # Estimate factor for searching (1 + epsilon)
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
max_continous_pages=5               # Max contiguous pages to read from disk
full_decode_volume=5                # Conservative pruning decode batches
partial_one_decode_volume=5         # Aggressive PQ decode (round 1)
partial_two_decode_volume=5         # Aggressive PQ decode (round 2)
submit_per_round=5                  # Buckets submitted per round

verbose=0                           # Verbose mode 0 = off
