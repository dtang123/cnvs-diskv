# Data files
base_filepath=/path/to/sift1b/bigann_base.ubin
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
search_nprobes=250,300,350          # Bucket number to search (s)
search_estimate_factors_partial=1.01,1.03,1.05
search_threads=16
cache_vectors=0
query_for_warm_up=0

# Recommended defaults
search_estimate_factor=1.1          # Estimate factor for searching (1 + epsilon)
search_estimate_factor_high_dim=1.1
search_prune_factor=5.0
max_continous_pages=5               # Max contiguous pages to read from disk
full_decode_volume=3                # Conservative pruning decode batches
partial_one_decode_volume=10        # Aggressive PQ decode (round 1)
partial_two_decode_volume=8         # Aggressive PQ decode (round 2)
submit_per_round=10                 # Buckets submitted per round

verbose=0                           # Verbose mode 0 = off

# Hybrid-specific configuration
# Split partitions across disks: first disk1_count partitions use disk1 paths, remaining use disk2.
disk1_count=5

# Disk 1 base paths
search_disk1_disk_store_path=/path/to/build/disk/store
search_disk1_centroid_index_path=/path/to/build/centroid/index

# Disk 2 base paths
search_disk2_disk_store_path=/path/to/search/disk/store
search_disk2_centroid_index_path=/path/to/search/centroid/index

############################
# --- Key addition: per-disk load parameters ---
# (Used by load_indices; falls back to global search_* if unset)
############################
# Disk 1 specific
search1_estimate_factor=1.1
search1_estimate_factor_high_dim=1.1
search1_prune_factor=5
search1_top=2
search1_replicas=2
search1_max_continous_pages=10
search1_full_decode_volume=5
search1_partial_one_decode_volume=10
search1_partial_two_decode_volume=8
search1_submit_per_round=10

# Disk 2 specific
search2_estimate_factor=1.1
search2_estimate_factor_high_dim=1.1
search2_prune_factor=5
search2_top=25
search2_replicas=2
search2_max_continous_pages=6
search2_full_decode_volume=8
search2_partial_one_decode_volume=8
search2_partial_two_decode_volume=8
search2_submit_per_round=10

# Optional per-disk overrides
# search1_nprobe=80
# search1_estimate_factor_partial=0.99

# search2_nprobe=60
# search2_estimate_factor_partial=1.01
