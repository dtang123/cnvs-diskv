# Data files
base_filepath=/path/to/gist_base.bin
query_filepath=/path/to/gist_query.bin
ground_truth_filepath=/path/to/gist_groundtruth.bin

# Data formats / types
dataset_fmt=fbin
vector_type=float
queryset_fmt=fbin
truthset_fmt=ibin

# Basic dataset properties
d=960
nb=1000000
nq=1000
ratio=10

# Partitions and paths
# do not change characters after last slash or rename index files
partitions=1
build_disk_store_path=/path/to/DiskV_index/gist1m
build_centroid_index_path=/path/to/DiskV_index/gist1m_centroid
search_disk_store_path=s3://minio/warehouse/diskv/gist1m
search_centroid_index_path=/dev/shm/diskv_metadata/gist1m_centroid
search_index_meta_path=/dev/shm/diskv_metadata/gist1m

# Build phase parameters
nlist=2000
m=48                                # 960/48=20 ✓ must divide evenly
nbits=8
replicas=2
shrink_replicas=1.15
build_estimate_factor=1.2
build_prune_factor=2
build_metric_type=L2
build_memory_graph_efb=75
build_memory_graph_efs=150
build_memory_graph_M=16
build_threads=16                     # adjust to your machine

# Search phase parameters
k=100
k_per_partition=100
search_top=0
search_nprobes=10
search_estimate_factors_partial=1.03
search_threads=4
cache_vectors=0
query_for_warm_up=0

search_estimate_factor=1.1
search_estimate_factor_high_dim=1.1
search_prune_factor=1.2
max_continous_pages=5
full_decode_volume=5
partial_one_decode_volume=5
partial_two_decode_volume=5
submit_per_round=5

verbose=1
