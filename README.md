# DiskV

Vector databases have attracted significant attention with the rapid adoption of large language models (LLMs). To improve the cost efficiency of vector search, several disk-based vector indexes, such as SPANN and DiskANN, have been proposed. However, these designs are mainly optimized for high-end, ultra-fast SSDs (e.g., NVMe and Optane) and exhibit poor performance on commodity storage devices, such as HDDs, cloud block storage, and low-end SSDs (e.g., SATA SSDs), which are widely used in practice due to cost and capacity considerations.

In this work, we present DiskV, the first disk-based vector index designed to achieve high search performance on commodity storage. The basic idea of DiskV builds upon a clustering-based indexing structure that partitions the dataset into large buckets, leveraging sequential access patterns to favor commodity storage. To improve performance, DiskV introduces a suite of novel optimizations, including **query-aware adaptive search**, **segment-based pruning**, and an **optimized asynchronous I/O**. Extensive experiments on billion-scale datasets show that DiskV substantially outperforms existing disk-based indexes on commodity storage (by up to **19X**). We also integrate DiskV into PostgreSQL, which demonstrates its practicality in a real database system. Overall, DiskV provides a more **practical and cost-efficient foundation** for supporting large-scale AI applications.

---



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



