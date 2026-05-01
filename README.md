# DiskV


Vector databases have attracted significant attention with the rapid adoption of large language models (LLMs). To improve the cost efficiency of vector search, several disk-based vector indexes, such as SPANN and DiskANN, have been proposed. However, these designs are mainly optimized for high-end, ultra-fast SSDs (e.g., NVMe and Optane) and exhibit poor performance on commodity storage devices, such as HDDs, cloud block storage, and low-end SSDs (e.g., SATA SSDs), which are widely used in practice due to cost and capacity considerations.

In this work, we present DiskV, the first disk-based vector index designed to achieve high search performance on commodity storage. The basic idea of DiskV builds upon a clustering-based indexing structure that partitions the dataset into large buckets, leveraging sequential access patterns to favor commodity storage. To improve performance, DiskV introduces a suite of novel optimizations, including **query-aware adaptive search**, **segment-based pruning**, and an **optimized asynchronous I/O**. Extensive experiments on billion-scale datasets show that DiskV substantially outperforms existing disk-based indexes on commodity storage (by up to **19X**). We also integrate DiskV into PostgreSQL, which demonstrates its practicality in a real database system. Overall, DiskV provides a more **practical and cost-efficient foundation** for supporting large-scale AI applications.

---


## Build and search file for Gist1M
```
# Data files
base_filepath=/hdd_root/tang627/GIST1M/gist_base.bin
query_filepath=/hdd_root/tang627/GIST1M/gist_query.bin
ground_truth_filepath=/hdd_root/tang627/GIST1M/gist_groundtruth.bin

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
partitions=1
build_disk_store_path=/home/tang627/DiskV_index/gist1m
build_centroid_index_path=/home/tang627/DiskV_index/gist1m_centroid
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
```
###Build command
```
./build/demos/demo_script 0 ./demos/dataset_gist1m.sh
```
## Index Search Setup
```
#!/bin/bash
# setup_diskv_minio.sh
# Source this file: source ~/setup_diskv_minio.sh

# ── MinIO / S3 credentials ────────────────────────────────────────────────────
export MINIO_ENDPOINT=http://127.0.0.1:19000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=password
export AWS_DEFAULT_REGION=us-east-2

# ── Configuration File Path ───────────────────────────────────────────────────
# This is the file the binary will read for search_estimate_factor, etc.
export DISKV_CONFIG_FILE=~/dataset_gist1m.sh

# ── Index paths (Optional if paths are inside the config file) ────────────────
export DISKV_INDEX_FILE=/dev/shm/diskv_metadata/gist1m_0.index
export DISKV_CLUSTERED_PATH=s3://warehouse/diskv/gist1m_0.clustered
export DISKV_CENTROID_PATH=/dev/shm/diskv_metadata/gist1m_centroid_0

# ── Query / ground truth ──────────────────────────────────────────────────────
export DISKV_QUERY_FILE=/hdd_root/tang627/GIST1M/gist_query.bin
export DISKV_GT_FILE=/hdd_root/tang627/GIST1M/gist_groundtruth.bin

# ── Search binary ─────────────────────────────────────────────────────────────
export DISKV_BIN=~/DiskV_VLDB26/build/demos/run_diskv_search

# ── Runtime linker path for AWS SDK ──────────────────────────────────────────
export LD_LIBRARY_PATH=/home/tang627/awssdk/lib:$LD_LIBRARY_PATH

# ── RAM disk setup ────────────────────────────────────────────────────────────
echo "Setting up RAM disk metadata..."
mkdir -p /dev/shm/diskv_metadata
if [[ ! -f /dev/shm/diskv_metadata/gist1m_0.index ]]; then
    cp ~/DiskV_index/gist1m_0.index    /dev/shm/diskv_metadata/
    echo "  Copied gist1m_0.index"
else
    echo "  gist1m_0.index already in RAM disk"
fi
if [[ ! -f /dev/shm/diskv_metadata/gist1m_centroid_0 ]]; then
    cp ~/DiskV_index/gist1m_centroid_0 /dev/shm/diskv_metadata/
    echo "  Copied gist1m_centroid_0"
else
    echo "  gist1m_centroid_0 already in RAM disk"
fi

# ── Verify MinIO ──────────────────────────────────────────────────────────────
echo "Checking MinIO..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    ${MINIO_ENDPOINT}/minio/health/live || echo "000")
if [[ "${HTTP_CODE}" != "200" ]]; then
    echo "  WARNING: MinIO not reachable (HTTP ${HTTP_CODE})"
    echo "  Run: cd ~/my-lab && docker compose up -d"
else
    echo "  MinIO OK (HTTP ${HTTP_CODE})"
fi

# ── Verify index in MinIO ─────────────────────────────────────────────────────
if ! ~/mc stat minio/warehouse/diskv/gist1m_0.clustered > /dev/null 2>&1; then
    echo "  WARNING: gist1m_0.clustered not found in MinIO"
    echo "  Run: ~/mc cp ~/DiskV_index/gist1m_0.clustered minio/warehouse/diskv/gist1m_0.clustered"
else
    echo "  Index file found in MinIO"
fi

# ── Print run command ─────────────────────────────────────────────────────────
echo ""
echo "Environment ready. Run the search using the config file:"
echo ""
echo "  \${DISKV_BIN} \${DISKV_CONFIG_FILE}"
echo ""
echo "Note: Ensure \${DISKV_CONFIG_FILE} has correct paths for S3 and metadata."
```
## Installation

### Dependencies

```
Basic requirements:

- A C++17 compiler (with OpenMP version 2.0 or higher)
- A BLAS implementation (Intel MKL is strongly recommended for best performance on Intel machines)
```

### Compilation

You can quickly build DiskV with the following commands:

```
cd DiskV

cmake -DFAISS_OPT_LEVEL=generic -DBUILD_SHARED_LIBS=ON \
      -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=ON \
      -DCMAKE_BUILD_TYPE=Release -B build

make -C build install
```

### Basic Testing

```
make -C build demo_ivfpq_indexing
./build/demos/demo_ivfpq_indexing
```

------

## Experiments

### Usage

DiskV provides two demo scripts: `demo_script.cpp` and `demo_script_hybrid.cpp`.
 The hybrid version supports **searching data stored on two different disks**.

#### Case 1: Search on a single storage device

First, compile the demo script:

```
make -C build demo_script
```

Then run the build and search phases with a parameter script:

```
# Build index
./build/demos/demo_script 0 ./demos/dataset_sift100m.sh

# Search
./build/demos/demo_script 1 ./demos/dataset_sift100m.sh
```

------

#### Case 2: Search across two storage devices

First, compile the hybrid version:

```
make -C build demo_script_hybrid
```

Then run the build and search phases:

```
# Build index
./build/demos/demo_script_hybrid 0 ./demos/dataset_sift1b_hybrid.sh

# Search
./build/demos/demo_script_hybrid 1 ./demos/dataset_sift1b_hybrid.sh
```

------

### Parameters

Recommended parameters for the **SIFT1B**, **DEEP1B**, and **Text2Image1B** datasets are listed below:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `partition` | Number of segments | `10` |
| `nlist` | Number of buckets in each segment | `200,000` (for all datasets) |
| `ratio` | Sampling ratio used during IVF_PQ clustering | `10` (i.e., sample 10%) |
| `m` | Number of sub-vectors in IVF_PQ | `16` (SIFT1B); `24` (DEEP1B); `100` (Text2Image1B) |
| `c_pq` | Number of clusters per sub-vector space | `256` |
| `M` | Maximum neighbor count for the centroid graph | `16` |
| `efb` | Priority queue length during centroid-graph construction | `75` |
| `efs` | Priority queue length during centroid-graph search | `150` |
| `N` | Number of lists to probe during search | `200` |
| `f` | Number of lists to search with sequential I/O | Auto-adjusted (typically `3` for SSD/GP3, `30` for HDD) |
| `ε` (epsilon) | Estimated filtering factor | `0.03` |


------

### Experiments

#### Datasets

We evaluate DiskV on three billion-scale datasets:

| Dataset | Type | Distance | #Dim. | #Vectors | #Queries |
|---------|------|----------|-------|----------|----------|
| SIFT1B | uint8 | L2 | 128 | 1,000,000,000 | 10,000 |
| DEEP1B | float | L2 | 96 | 1,000,000,000 | 10,000 |
| Text2Image1B  | float | IP | 200 | 1,000,000,000 | 100,000 |
| Cohere10M  | float | L2 | 768 | 10,000,000 | 1,000 |

#### Storage Performance Statistics

We evaluate DiskV on various storage devices.
`Fio` is used to test the actual IOPS and throughput for both the local disk storage and cloud disk volumes. 
We set the test in a vector search scenario by employing an iodepth of 64 with 16 concurrent jobs. The program runs with 4KB and 128KB block-size respectively to measure the random and sequential I/O performance during a realistic 60-second workload on a 1GB data file. 
Below are the details:

<table>
<tr>
<th style="min-width: 150px;">Disk Storage</th>
<th>Preset IOPS</th>
<th>Tested IOPS</th>
<th>Preset Bandwidth (MB/s)</th>
<th>Tested Bandwidth (MB/s)</th>
</tr>
<tr><td>GP3_1 (Cloud SSD)</td><td>3,000</td><td>3,121</td><td>125</td><td>125</td></tr>
<tr><td>GP3_2 (Cloud SSD)</td><td>7,000</td><td>7,089</td><td>125</td><td>125</td></tr>
<tr><td>GP3_3 (Cloud SSD)</td><td>11,000</td><td>11,173</td><td>125</td><td>125</td></tr>
<tr><td>IO2 (Cloud SSD)</td><td>18,750</td><td>18,912</td><td>N/A</td><td>576</td></tr>
<tr><td>Cloud HDD</td><td>N/A</td><td>203</td><td>N/A</td><td>27</td></tr>
<tr><td>Local SSD</td><td>N/A</td><td>76,352</td><td>N/A</td><td>548</td></tr>
<tr><td>Local HDD</td><td>N/A</td><td>339</td><td>N/A</td><td>68</td></tr>
</table>

#### Results

All billion datasets has been divided into 10 segments. The build time is defined as the end-to-end duration from the start of the raw dataset to the completion of building all indexes required for searching. We record the on-disk footprint of each resulting index as index size. 

The query sets for each segment are the same as the query sets from the whole dataset. Query is executed against all segments, and throughput is reported as end-to-end QPS (queries per second).

##### Build time and index size :
<img src="figures/combined_graph.png" alt="buildtime_indexsize" width="400">

##### AWS GP3 Cloud SSD for SIFT1B :

<img src="figures/GP3_sift1b.png" alt="GP3 SIFT1B Performance" width="700">

##### AWS GP3 Cloud SSD for DEEP1B :

<img src="figures/GP3_deep1B.png" alt="GP3 DEEP1B Performance" width="700">

##### AWS GP3 Cloud SSD for Cohere10M :

<img src="figures/GP3_cohere10M.png" alt="GP3 Cohere10M Performance" width="700">

##### AWS IO2 Cloud SSD :
<img src="figures/cloud_IO2.png" alt="Cloud IO2 Performance" width="700">

##### AWS Cloud HDD :
<img src="figures/cloud_hdd.png" alt="Cloud HDD Performance" width="700">

##### Local SSD :

<img src="figures/Local_SSD.png" alt="Local SSD Performance" width="1000">

##### Local HDD :

<img src="figures/Local_HDD.png" alt="Local HDD Performance" width="1000">

##### Cache Performance on Local SSD for SIFT1B:
<img src="figures/cache.png" alt="Cache Performance" width="400">



