# Data files
base_filepath=/path/to/sift1b/bigann_100m_base.ubin
query_filepath=/path/to/sift1b/bigann_query.ubin
ground_truth_filepath=/path/to/sift1b/gnd/idx_100M.ibin

# Data formats / types
dataset_fmt=u8bin                  # bvecs/ivecs/fbin/ibin/u8bin
vector_type=uint8                  # uint8/int/float
queryset_fmt=u8bin                 # bvecs/ivecs/fbin/ibin/u8bin
truthset_fmt=ibin                  # ivecs/ibin

# Basic dataset properties
d=128
nb=100000000
nq=10000
ratio=10                            # sample ratio (100/ratio)% data used to build the index

# Partitions and paths
partitions=1
build_disk_store_path=/path/to/build/disk/store
build_centroid_index_path=/path/to/build/centroid/index
search_disk_store_path=/path/to/search/disk/store
search_centroid_index_path=/path/to/search/centroid/index

# Build phase parameters
nlist=200000                       # IVF cluster count
m=16                                # PQ sub-quantizers
nbits=8                             # PQ bit width
replicas=2                          # Replicas for building
shrink_replicas=1.15                # Shrink replicas by remove some points
build_estimate_factor=1.2           # Estimate factor for building (1 + episilon)
build_prune_factor=2                # Prune factor for building
build_metric_type=L2                # IP/L2
build_memory_graph_efb=75           # HNSW in-memory graph efConstruction
build_memory_graph_efs=150          # HNSW in-memory graph efSearch
build_memory_graph_M=16             # HNSW graph degree (M) for building
build_threads=128                   # Threads for building


# Search phase parameters
k=100                               # candidates to retrieve
k_per_partition=100                 # retrieve k per partition candidates from each partition                                
search_top=2                        # Conservative Pruning (t)
search_nprobes=50,80,120,160,200    # Bucket number to search (s) 
search_estimate_factors_partial=1.03 # Partial estimate factor for searching (1 + episilon)
search_threads=16
cache_vectors=0
query_for_warm_up=0

# Parameters below are recommended to be left as default.
                                    # Usually, search_estimate_factor and search_estimate_factors_partial are set to the same value.
                                    # But we only need to set partial factors to get different performance.
search_estimate_factor=1.1          # Estimate factor for searching (1 + episilon)
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
max_continous_pages=5               # Max continous pages to read from disk
full_decode_volume=5                # How many batches pq lookup can be decoded in Conservative Pruning
partial_one_decode_volume=5         # How many batches pq lookup can be decoded in first round of Aggressive PQ Decoding  
partial_two_decode_volume=5         # How many batches pq lookup can be decoded in second round of Aggressive PQ Decoding
submit_per_round=5                  # How many buckets to submit each round 


verbose=0                           # Verbose mode  0 for not give any log
