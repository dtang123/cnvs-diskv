#include <faiss/IndexDiskV.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_set>
#include <cinttypes>
#include <cstdio>
#include <limits>

#include <fstream>
#include <unistd.h>  
#include <fcntl.h>   
#include <sys/types.h> 
#include <sys/stat.h>  
#include <cerrno> 
#include <stdexcept>  
#include <sys/mman.h>

#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <faiss/index_io.h>
#include <faiss/tsl/robin_map.h>


#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/DiskIOProcessor.h>

#include <faiss/impl/code_distance/code_distance.h>

#include <iostream>
#include <string>

#define USING_ASYNC
//#define USING SYNC



namespace faiss{

IndexDiskV::IndexDiskV(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        size_t top,
        float estimate_factor,
        float prune_factor,
        const std::string& diskPath,
        const std::string& valueType,
        MetricType metric):
    IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, metric),
                top(top),
                estimate_factor(estimate_factor),
                prune_factor(prune_factor),
                disk_path(diskPath),
                build_memory_graph_ef_construction(40),
                build_memory_graph_ef_search(16),
                build_memory_graph_M(16),
                disk_vector_offset(d * sizeof(float)),
                valueType(valueType) {
    estimate_factor_partial = estimate_factor;

    std::cout << valueType << " " << this->valueType << std::endl;
    if(valueType!="float" && valueType!= "uint8" && valueType!= "int16"){

        FAISS_THROW_FMT("Unsupported type %s", valueType.c_str());
    }

    // We want a new type of InvertedList here.
    if (own_invlists && invlists) {
        delete invlists;
    }
    // Assign a new ClusteredArrayInvertedLists instance
    invlists = new ClusteredArrayInvertedLists(nlist, code_size);
    own_invlists = true;  // Ensure the IndexIVF class takes ownership
    this->aligned_cluster_info = nullptr;
    clusters = nullptr;
    len = nullptr;
}

IndexDiskV::IndexDiskV() {}

IndexDiskV::~IndexDiskV() {
    if(clusters != nullptr)
        delete[] clusters;
    if(len != nullptr)
        delete[] len;
    if(aligned_cluster_info!=nullptr)
        delete[] aligned_cluster_info;
    if(cached_list_info != nullptr)
        delete[] cached_list_info;
    
}

void IndexDiskV::train_graph(){
    std::cout << "The index is of type Disk.\n";
    IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
    if (flat_quantizer != nullptr) {
        faiss::IndexHNSWFlat index(d, build_memory_graph_M, this->metric_type);
        index.hnsw.efConstruction = build_memory_graph_ef_construction;
        index.hnsw.efSearch = build_memory_graph_ef_search;
        index.add(quantizer->ntotal, flat_quantizer->get_xb());
        faiss::write_index(&index, this->centroid_index_path.c_str());
        std::cout << "Output centroid index.\n";
        load_hnsw_centroid_index();
    } else {
        std::cerr << "Quantizer is not an IndexFlat." << std::endl;
    }
}

void IndexDiskV::load_hnsw_centroid_index(){
    if (centroid_index_path.empty()) {
        throw std::runtime_error("Centroid index path is not set.");
    }
    faiss::Index* loaded_index = faiss::read_index(centroid_index_path.c_str());
    centroid_index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded_index);
    if (centroid_index == nullptr) {
        throw std::runtime_error("Failed to cast the loaded index to faiss::IndexHNSW.");
    }
}

void IndexDiskV::set_memory_graph_build_params(
        int ef_construction, int ef_search, int graph_M) {
    build_memory_graph_ef_construction = ef_construction;
    build_memory_graph_ef_search = ef_search;
    build_memory_graph_M = graph_M;
}

namespace{
double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2, size_t d) {
    double dot = 0.0;
    for (size_t i = 0; i < d; i++) {
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

// Function to calculate the norm (magnitude) of a vector
double vectorNorm(const std::vector<double>& vec, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; i++) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}
// Function to calculate the residual vector (difference between two vectors)
std::vector<double> calculateResidual(const std::vector<double>& vec, const std::vector<double>& centroid, size_t d) {
    std::vector<double> residual(d);
    for (size_t i = 0; i < d; i++) {
        residual[i] = vec[i] - centroid[i];
    }
    return residual;
}
// Function to retrieve a centroid vector from centroidData given an index
std::vector<double> getCentroidVector(const float* centroidData, idx_t centroidIdx, size_t d) {
    std::vector<double> centroid(d);
    for (size_t i = 0; i < d; i++) {
        centroid[i] = static_cast<double>(centroidData[centroidIdx * d + i]);
    }
    return centroid;
}
// Function to retrieve a vector from xb given an index
std::vector<double> getVector(const float* xb, size_t vecIdx, size_t d) {
    std::vector<double> vec(d);
    for (size_t i = 0; i < d; i++) {
        vec[i] = static_cast<double>(xb[vecIdx * d + i]);
    }
    return vec;
}
// Select a set of centroid assignments while encouraging diversity among residual directions.
void selectDiverseCentroids(const float* xb,
          const float* centroidData,
          const idx_t* coarse_idx,  // [nb * k], the candidate centroid IDs
          idx_t* assign,           // [nb * num_centroid] output (chosen IDs)
          size_t nb,               // number of vectors
          size_t d,                // dimension
          size_t k,                // how many centroid candidates per vector
          size_t num_centroid)     // how many we actually pick
{
    if (num_centroid > k) {
        fprintf(stderr, 
                "ERROR: num_centroid (%zu) cannot exceed k (%zu)\n", 
                num_centroid, k);
        return;
    }

#pragma omp parallel for
    for (size_t vecIdx = 0; vecIdx < nb; vecIdx++) 
    {
        std::vector<double> vec = getVector(xb, vecIdx, d);
        double minDist = std::numeric_limits<double>::max();
        idx_t bestCentroidID = 0;

        for (size_t i = 0; i < k; i++) {
            idx_t cID = coarse_idx[vecIdx * k + i];
            std::vector<double> cvec = getCentroidVector(centroidData, cID, d);

            std::vector<double> tmpResidual = calculateResidual(vec, cvec, d);
            double dist = dotProduct(tmpResidual, tmpResidual, d);  
            if (dist < minDist) {
                minDist = dist;
                bestCentroidID = cID;
            }
        }

        assign[vecIdx * num_centroid + 0] = bestCentroidID;

        std::vector<double> primaryCentroid = getCentroidVector(centroidData, bestCentroidID, d);
        std::vector<double> primaryResidual = calculateResidual(vec, primaryCentroid, d);
        double primaryResidualNorm = vectorNorm(primaryResidual, d);

        std::unordered_set<idx_t> used;
        used.insert(bestCentroidID);

        for (size_t pick = 1; pick < num_centroid; pick++) {
            double minCosSim = std::numeric_limits<double>::max();
            idx_t bestSecID  = bestCentroidID;

            for (size_t i = 0; i < k; i++) {
                idx_t cID = coarse_idx[vecIdx * k + i];
                if (used.count(cID) > 0) {
                    continue;
                }

                std::vector<double> cvec = getCentroidVector(centroidData, cID, d);
                std::vector<double> currentResidual = calculateResidual(vec, cvec, d);
                double currentResidualNorm = vectorNorm(currentResidual, d);

                double cosSim = dotProduct(primaryResidual, currentResidual, d) / 
                                (primaryResidualNorm * currentResidualNorm + 1e-12);

                if (cosSim < minCosSim) {
                    minCosSim = cosSim;
                    bestSecID = cID;
                }
            }


            if (bestSecID == bestCentroidID && used.size() < k) {
                for (size_t i = 0; i < k; i++) {
                    idx_t cID = coarse_idx[vecIdx * k + i];
                    if (!used.count(cID)) {
                        bestSecID = cID;
                        break;
                    }
                }
            }

            assign[vecIdx * num_centroid + pick] = bestSecID;
            used.insert(bestSecID);
        }
    } 
}

}
namespace{

    void shrink_assignments(float* distances,
                            idx_t* assign,
                            size_t k,
                            size_t n,
                            MetricType metric) {
        constexpr float kControl = 1.10f;

        if (k <= 1 || n == 0) {
            return;
        }

        std::cout << "Shrinking replica assignments." << std::endl;

        size_t shrinked_num = 0;

#pragma omp parallel for reduction(+:shrinked_num) schedule(dynamic)
        for (size_t i = 0; i < n; i++) {
            const size_t distance_base = i * k * 2;
            const size_t assign_base = i * k;

            for (size_t j = 1; j < k; j++) {
                const float previous = distances[distance_base + j - 1];
                const float current = distances[distance_base + j];
                bool should_shrink = false;

                switch (metric) {
                    case METRIC_L2:
                        should_shrink = current > kControl * previous;
                        break;
                    case METRIC_INNER_PRODUCT:
                        should_shrink = current < previous / kControl;
                        break;
                    default:
                        should_shrink = false;
                        break;
                }

                if (should_shrink) {
                    const size_t removed = k - j;
                    for (size_t r = j; r < k; r++) {
                        assign[assign_base + r] = -1;
                    }
                    shrinked_num += removed;
                    break;
                }
            }
        }

        const size_t total = n * k;
        const double ratio = total == 0
                ? 0.0
                : static_cast<double>(shrinked_num) * 100.0 / static_cast<double>(total);

        std::cout << "total:" << total
                  << "  shrinked:" << shrinked_num
                  << "  ratio:" << ratio << "%\n";
    }

}


void IndexDiskV::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    idx_t k = this->assign_replicas;


    if(this->soaring && k > 1){
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k * 2]);
        printf("IndexDiskV::add_with_ids::k=%ld\n",k);
        float* D = new float[k * n * 2];

        if (n * k * 2 < 5000) 
        {
            quantizer->search(n, x, k * 2, D, coarse_idx.get());
        } else {
            printf("IndexDiskV::add_with_ids::hnsw search, n=%ld, k=%ld \n", n, k * 2);
            centroid_index->hnsw.efSearch = 400;
            centroid_index->search(n, x, k * 2, D, coarse_idx.get());
        }
        if (k != 1) {
            printf("Using centroid diversification\n");
            std::unique_ptr<idx_t[]> assign(new idx_t[n * k]);
            IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
            float* centroidData = flat_quantizer->get_xb();
            selectDiverseCentroids(x, centroidData, coarse_idx.get(), assign.get(), n, d, k * 2, k);

            shrink_assignments(D, assign.get(), k, n, this->metric_type);  // k = 2, D' k = 4

            add_core(n, x, xids, assign.get());
            delete[] D;
            return;
        }     
        delete[] D;
        add_core(n, x, xids, coarse_idx.get());
    }else{
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k]);
        printf("IndexDiskV::add_with_ids::k=%ld\n",k);
        float* D = new float[k * n];
        if (n * k * 2 < 5000) 
        {
            quantizer->search(n, x, k, D, coarse_idx.get());
        } else {
            printf("IndexDiskV::add_with_ids::hnsw search, n=%ld, k=%ld \n", n, k);
            centroid_index->hnsw.efSearch = 400;
            centroid_index->search(n, x, k, D, coarse_idx.get());
        }

        delete[] D;
        add_core(n, x, xids, coarse_idx.get());
    }
}

void IndexDiskV::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    add_core_o(n, x, xids, nullptr, coarse_idx, inverted_list_context);
    initial_location(n, x);
}

//#define BUILD_IN_MEMORY

#ifdef BUILD_IN_MEMORY
namespace{
    void add_original_data(const float* data, ArrayInvertedLists* build_invlists, ClusteredArrayInvertedLists* c_array_invlists, size_t old_total, size_t d) {

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < c_array_invlists->nlist; ++i) {
            size_t new_count = c_array_invlists->ids[i].size();
            size_t old_count = build_invlists->ids[i].size();

            for (size_t j = old_count; j < new_count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                const float* vector_to_add = &data[(id-old_total) * d];
                build_invlists->add_entry(i, id, (uint8_t*)vector_to_add);  // Assume ArrayInvertedLists can append vectors via add_entry
            }
        }
    }

}
#endif

void IndexDiskV::initial_location(idx_t n, const float* data) {
    if (!invlists) {
        throw std::runtime_error("invlists is not initialized.");
    }

    ClusteredArrayInvertedLists* array_invlists = dynamic_cast<ClusteredArrayInvertedLists*>(invlists);
    if (!array_invlists) {
        throw std::runtime_error("invlists is not of type ClusteredArrayInvertedLists.");
    }

    size_t* tmp_clusters = nullptr;
    size_t* tmp_len = nullptr;
    if(clusters == nullptr && len == nullptr){
        this->aligned_cluster_info = new Aligned_Cluster_Info[nlist];
        clusters = new size_t[nlist];
        len = new size_t[nlist];
    }else{
        tmp_clusters = new size_t[nlist];
        tmp_len = new size_t[nlist];
        for(size_t i = 0; i < nlist; ++i){
            tmp_clusters[i] = clusters[i];
            tmp_len[i] = len[i];
        }
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < nlist; ++i) {
        clusters[i] = current_offset;
        len[i] = array_invlists->ids[i].size();
        current_offset += len[i];
        // update maps
        array_invlists->add_maps(i, len[i]);
    }
    if(verbose){
        printf("Cluster info initialized!");
    }

    std::unique_ptr<DiskIOProcessor> io_processor(get_DiskIOBuildProcessor());
    ClusteredArrayInvertedLists* c_array_invlists = dynamic_cast<ClusteredArrayInvertedLists*>(invlists);
    if (!c_array_invlists) {
        throw std::runtime_error("invlists is not of type ArrayInvertedLists.");
    }
    bool do_in_list_cluster = false;
    bool pq_info_keep_in_disk_file = false;
    bool modify_ids_in_multi_index = false;

    this->actual_batch_num++;
    if(this->actual_batch_num == this->add_batch_num){
        if(this->in_list_clustering == true)
            do_in_list_cluster = true;
        pq_info_keep_in_disk_file = true;

        if(this->use_multi_ivf)
            modify_ids_in_multi_index = true;
    }
        
    std::cout << "batch:" << this->actual_batch_num << std::endl;
#ifndef BUILD_IN_MEMORY 
    if(io_processor->reorganize_vectors_2(n,
                                    data,
                                    tmp_clusters,
                                    tmp_len,
                                    clusters,
                                    len,
                                    aligned_cluster_info,
                                    nlist,
                                    c_array_invlists,
                                    do_in_list_cluster)){
        this->disk_path = disk_path + ".clustered";
        
    }
#else
    if(this->build_invlists == nullptr){
        this->build_invlists = new ArrayInvertedLists(nlist, d*sizeof(float));
    }
    add_original_data(data, build_invlists, c_array_invlists, this->ntotal-n, d);

    if(io_processor->reorganize_vectors_in_memory(n,
                                    data,
                                    tmp_clusters,
                                    tmp_len,
                                    clusters,
                                    len,
                                    aligned_cluster_info,
                                    nlist,
                                    c_array_invlists,
                                    build_invlists,
                                    do_in_list_cluster,
                                    keep_in_disk)){
        this->disk_path = disk_path + ".clustered";
    }
#endif


    if(reorganize_lists && do_in_list_cluster){
        std::cout << "original data deleted\n";
        delete[] this->xb;
        std::cout << "reorganizing list..........\n";
        auto time_start = std::chrono::high_resolution_clock::now();
        io_processor->reorganize_list(*quantizer, c_array_invlists, aligned_cluster_info, clusters, len, nlist);
        train_graph();
        auto time_end = std::chrono::high_resolution_clock::now();
        auto reorg_time = time_end - time_start;
        std::cout << "reorganize cost:" << reorg_time.count()/ 1000 << "\n";
    }

    if(modify_ids_in_multi_index){
        this->multi_ivf_modify_id();
    }

    if(select_lists && pq_info_keep_in_disk_file){
        std::cout << "pq_size = " << this->pq.code_size << "  idx_t = " << sizeof(faiss::idx_t)<< "  size_t = " << sizeof(size_t) << "\n";


        size_t entry_size = this->pq.code_size + sizeof(faiss::idx_t) + 0;
        this->aligned_inv_info = new Aligned_Invlist_Info[nlist];
        io_processor->organize_select_list(this->pq.code_size, entry_size, c_array_invlists, aligned_inv_info, nlist, this->disk_path);

    }

    if(tmp_clusters != nullptr){
        delete[] tmp_clusters;
        delete[] tmp_len;
    }

}


void IndexDiskV::reorganize_vectors(idx_t n, const float* data, size_t* old_clusters, size_t* old_len) {

    ArrayInvertedLists* array_invlists = dynamic_cast<ArrayInvertedLists*>(invlists);
    if (!array_invlists) {
        throw std::runtime_error("invlists is not of type ArrayInvertedLists.");
    }

    idx_t old_total = this->ntotal - n;

    if (old_clusters == nullptr && old_len == nullptr) {
        set_disk_write(disk_path);
        for (size_t i = 0; i < nlist; ++i) {
            size_t count = len[i];
            for (size_t j = 0; j < count; ++j) {
                idx_t id = array_invlists->ids[i][j];
                const float* vector = &data[id * d];
                disk_data_write.write(reinterpret_cast<const char*>(vector), d * sizeof(float));
            }
        }
        disk_data_write.close();
    } else {
        std::string tmp_disk = disk_path + ".tmp";
        int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
        if(file_result == 0)
            std::cout << "Success rename: " << tmp_disk << std::endl;
        else
            std::cout << "Fail: "<< tmp_disk << std::endl;
        std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
        if (!temp_disk_read.is_open()) {
            throw std::runtime_error("Failed to open temporary disk file for reading.");
        }

        set_disk_write(disk_path);

        for (size_t i = 0; i < nlist; ++i) {
            size_t old_offset = old_clusters[i];
            size_t old_count = old_len[i];

            std::vector<float> old_cluster(old_count * d);
            temp_disk_read.seekg(old_offset * d * sizeof(float), std::ios::beg);
            temp_disk_read.read(reinterpret_cast<char*>(old_cluster.data()), old_count * d * sizeof(float));
            disk_data_write.write(reinterpret_cast<const char*>(old_cluster.data()), old_count * d * sizeof(float));

            size_t count = len[i];
            for (size_t j = old_count; j < count; ++j) {
                idx_t id = array_invlists->ids[i][j];
                const float* vector = &data[(id - old_total) * d];
                disk_data_write.write(reinterpret_cast<const char*>(vector), d * sizeof(float));
            }
        }
        disk_data_write.close();
        temp_disk_read.close();

        std::remove(tmp_disk.c_str());
    }

    if (verbose) {
        printf("Vectors reorganized and written to %s\n", disk_path.c_str());
    }
}

// Method to set the disk path and open read stream
void IndexDiskV::set_disk_read(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_read.is_open()) {
        disk_data_read.close();
    }
    disk_data_read.open(disk_path, std::ios::binary);
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("IndexDiskV: Failed to open disk file for reading");
    }
}

// Method to set the disk path and open write stream
void IndexDiskV::set_disk_write(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_write.is_open()) {
        disk_data_write.close();
    }
    disk_data_write.open(disk_path, std::ios::binary);
    if (!disk_data_write.is_open()) {
        throw std::runtime_error("IndexDiskV: Failed to open disk file for writing");
    }
}

void IndexDiskV::load_from_offset(size_t list_no, size_t offset, float* original_vector) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }
    assert(offset < len[list_no]);
    size_t global_offset = (clusters[list_no] + offset) * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);
    disk_data_read.read(reinterpret_cast<char*>(original_vector), d * sizeof(float));
    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vector from disk.");
    }
}

void IndexDiskV::load_clusters(size_t list_no, float* original_vectors) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }

    size_t global_offset = clusters[list_no] * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);
    disk_data_read.read(reinterpret_cast<char*>(original_vectors), d * sizeof(float) * len[list_no]);

    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vectors from disk.");
    }
}

namespace{
     int sort_coarse(
        std::vector<idx_t>& listno,            // input: cluster ids
        std::vector<size_t>& sorted_listno,    // output: sorted cluster ids based on frequency
        size_t* lens,                          // input: array of cluster sizes (number of vectors in each cluster)
        size_t nlist,                          // input: maximum number of clusters to cache
        size_t nvec)                           // input: maximum number of vectors to cache
    {
        std::map<idx_t, size_t> freq_map;
        for (const auto& id : listno) {
            freq_map[id]++;
        }

        std::vector<std::pair<idx_t, size_t>> freq_pairs(freq_map.begin(), freq_map.end());

        std::sort(freq_pairs.begin(), freq_pairs.end(),
            [](const std::pair<idx_t, size_t>& a, const std::pair<idx_t, size_t>& b) {
                return a.second > b.second;
            });

        size_t total_vecs = 0;

        for (const auto& pair : freq_pairs) {
            idx_t cluster_id = pair.first;
            if (nlist != static_cast<size_t>(-1) && sorted_listno.size() >= nlist) {
                break;
            }

            if (nvec != static_cast<size_t>(-1)) {
                total_vecs += lens[cluster_id];
                if (total_vecs >= nvec) {
                    sorted_listno.push_back(cluster_id);
                    break;
                }
            }
            sorted_listno.push_back(cluster_id);
        }
        return sorted_listno.size();
    }

    int warm_up(DiskInvertedListHolder& holder, std::vector<size_t>& indice){
        holder.warm_up(indice);
        return 0;
    }

    int warm_up(DiskVectorHolder& holder, std::vector<size_t>& indice){
        holder.warm_up(indice);
        return 0;
    }

}

int IndexDiskV::warmUpListCache(size_t n, float* x, size_t w_nprobe, size_t warm_list){
    std::vector<idx_t> idx(n * w_nprobe);
    std::vector<float> coarse_dis(n * w_nprobe);
    std::vector<size_t> sorted_idx;
    
    // Search to get the nearest cluster IDs
    // quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);
    if (warm_list != 0){
        quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);
        sort_coarse(idx, sorted_idx, this->len, warm_list, static_cast<size_t>(-1));
    }
    // sort_coarse(idx, sorted_idx, this->len, warm_list, static_cast<size_t>(-1));

    size_t code_size = get_code_size();

    diskInvertedListHolder.set_holder(disk_path, nlist, code_size, aligned_cluster_info);

    // for(int i = 0; i< warm_list;i++){
    //     std::cout << "idx:"<<sorted_idx.data()[i] << "  length:" << this->len[sorted_idx.data()[i]] << std::endl;
    // }

    warm_up(diskInvertedListHolder, sorted_idx);

    return warm_list;
}


int IndexDiskV::warmUpVectorCache(size_t n, float* x, size_t w_nprobe, size_t k, size_t nvec, float efp, size_t n_threads){
    size_t cache_top = 3;
    std::cout << "cache search threads num:" << n_threads << std::endl;
    initializeVectorCollector(n_threads);
    size_t vector_code_size = get_code_size();
    diskVectorHolder.set_holder(disk_path, nlist, vector_code_size, aligned_cluster_info);
    vector_cache_setting_mode = true;
    size_t top_set = this->top;
    size_t nprobe_set = this->nprobe;
    float fpartia_set = this->estimate_factor_partial;
    this->estimate_factor_partial = efp;
    this->top = cache_top;
    this->nprobe = w_nprobe;
    if(nvec != 0)
        this->search(n, x, k, nullptr,nullptr);

    std::vector<std::vector<VectorPair>> vec_freq= finalizeVectorCollector(n_threads);

    this->estimate_factor_partial = fpartia_set;
    this->nprobe = nprobe_set;
    this->top = top_set;
    vector_cache_setting_mode = false;

    std::vector<size_t> sorted_vecs = diskVectorHolder.sort_vectors_to_cache(vec_freq, nvec);
    
    warm_up(diskVectorHolder, sorted_vecs);

    return sorted_vecs.size();
}



int IndexDiskV::warmUpVectorCacheDp(size_t n, float* x, size_t w_nprobe, size_t k, size_t nvec, float efp, size_t n_threads){
    size_t cache_top = 3;
    std::cout << "cache search threads num:" << n_threads << std::endl;
    initializeVectorCollector(n_threads);
    size_t vector_code_size = get_code_size();

    size_t capacity = nvec*1;
    this->diskVectorHolder_dp = new DiskVectorHolder_dp(capacity, vector_code_size, 
        this->nlist, this->disk_path, this->aligned_cluster_info);

    vector_cache_dp_setting_mode = true;
    size_t top_set = this->top;
    size_t nprobe_set = this->nprobe;
    float fpartia_set = this->estimate_factor_partial;
    bool runtime_cache_set = this->runtime_cache; 

    this->estimate_factor_partial = efp;
    this->top = cache_top;
    this->nprobe = w_nprobe;
    this->runtime_cache = false;
    if(nvec != 0)
        this->search(n, x, k, nullptr,nullptr);

    std::vector<std::vector<VectorPair>> vec_freq= finalizeVectorCollector(n_threads);
    
    // restore settings
    this->estimate_factor_partial = fpartia_set;
    this->nprobe = nprobe_set;
    this->top = top_set;
    vector_cache_dp_setting_mode = false;
    this->runtime_cache = runtime_cache_set;

    std::vector<WarmupVecInfo> sorted_vecs = diskVectorHolder_dp->sort_vectors_to_cache(vec_freq, nvec);
    diskVectorHolder_dp->warm_up_2(sorted_vecs);

    if(this->runtime_cache){
        this->local_merge_size = 100;
        diskVectorHolder_dp->local_merge_size = this->local_merge_size;

        this->result_cached_queries.resize(n_threads);

        this->initializeLocalVectorCaches(n_threads, 10000);
    }

    return diskVectorHolder_dp->size;
}


int IndexDiskV::warmUpVectorCacheShard(size_t n, float* x, size_t w_nprobe, size_t k,size_t nvec,float efp, size_t n_threads, size_t n_shard){
    size_t cache_top = 3;
    initializeVectorCollector(n_threads);
    size_t vector_code_size = get_code_size();

    size_t capacity = nvec*1;
 
    vector_cache_dp_setting_mode = true; 
    size_t top_set = this->top;
    size_t nprobe_set = this->nprobe;
    float fpartia_set = this->estimate_factor_partial;
    bool runtime_shard_update_set = this->runtime_shard_update; 
    bool runtime_cache_set = this->runtime_cache; 
    bool batch_shard_update_set = this->batch_shard_update;
    bool cache_on_buffer_set = this->cache_on_buffer;

    this->estimate_factor_partial = efp;
    this->top = cache_top;
    this->nprobe = w_nprobe;
    this->runtime_shard_update = false;
    this->runtime_cache = false;
    this->batch_shard_update = false;
    this->cache_on_buffer = false;
    if(nvec != 0)
        this->search(n, x, k, nullptr,nullptr);

    this->diskVectorHolder_shard = new DiskVectorHolder_shard(n_shard, vector_code_size, capacity, 
                                    this->nlist, this->disk_path, this->aligned_cluster_info);
    std::vector<std::vector<VectorPair>> vec_freq= finalizeVectorCollector(n_threads);

    // restore settings
    this->estimate_factor_partial = fpartia_set;
    this->nprobe = nprobe_set;
    this->top = top_set;
    vector_cache_dp_setting_mode = false;
    this->runtime_shard_update = runtime_shard_update_set;
    this->runtime_cache = runtime_cache_set;
    this->batch_shard_update = batch_shard_update_set;
    this->cache_on_buffer = cache_on_buffer_set;

    std::vector<WarmupVecInfo> sorted_vecs = diskVectorHolder_shard->sort_vectors_to_cache(vec_freq, nvec);
    diskVectorHolder_shard->warm_up_2(sorted_vecs);
    if(this->runtime_cache)
    {
        if(cache_on_buffer){
            this->initializeLocalVectorCaches(n_threads, cache_buffer_size);
        }else{
            this->result_cached_queries.resize(n_threads);
            this->initializeLocalVectorCaches(n_threads, 1000);
        }
    }
    return diskVectorHolder_shard->get_global_size();
}

int IndexDiskV::merge_cache_local_to_global() const{
    using LocalCacheMap = std::unordered_map<idx_t, CacheNode*>;

    std::unordered_map<idx_t, std::pair<float, std::vector<size_t>>> total_local_map;

    size_t local_merge_size = diskVectorHolder_dp->local_merge_size * local_cache_submit_ratio;
    int merged_vector = 0;

    for (size_t i = 0; i < this->result_cached_queries.size(); i++) {
        faiss::DiskVectorHolder_local* local_cache = this->diskVectorHolders_local[i].get();
        std::shared_lock<std::shared_mutex> rlock(local_cache->local_mutex);
        for (const auto& [id, node] : local_cache->cache_map) {
            auto it = total_local_map.find(id);
            if (it == total_local_map.end()) {
                total_local_map[id] = std::make_pair(node->freq, std::vector<size_t>{i});
            } else {
                it->second.first += node->freq;
                it->second.second.push_back(i);
            }
        }
        this->result_cached_queries[i] = 0;
    }

    std::vector<std::pair<idx_t, std::pair<float, std::vector<size_t>>>> sorted_entries(
        total_local_map.begin(), total_local_map.end());

    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const auto& a, const auto& b) {
                  return a.second.first > b.second.first;
              });

    size_t limit = std::min(local_merge_size, sorted_entries.size());
    for (size_t i = 0; i < limit; i++) {
        idx_t id = sorted_entries[i].first;
        float local_freq_sum = sorted_entries[i].second.first;
        const std::vector<size_t>& sources = sorted_entries[i].second.second;

        const uint8_t* data_ptr = nullptr;
        for (size_t src_idx : sources) {
            faiss::DiskVectorHolder_local* local = this->diskVectorHolders_local[src_idx].get();
            auto it = local->cache_map.find(id);
            if (it != local->cache_map.end()) {
                data_ptr = it->second->data.data();
                break;
            }
        }

        if (!data_ptr) continue;

        float new_freq = local_freq_sum / static_cast<float>(local_merge_size);
        {
            std::unique_lock<std::shared_mutex> lock(diskVectorHolder_dp->global_mutex);
            auto it = diskVectorHolder_dp->cache_map.find(id);

            if (it == diskVectorHolder_dp->cache_map.end()) {
                diskVectorHolder_dp->insert_cache(id, data_ptr, new_freq);
                
            } else {
                CacheNode* dp_node = it->second;
                diskVectorHolder_dp->move_to_head(dp_node);
                
            }
            merged_vector++;
        }
    }
    std::cout << "Submitted: " << merged_vector << " Vectors" << std::endl;
    return merged_vector;
}

int IndexDiskV::warmUpPageCache(size_t n, float* x, size_t w_nprobe, size_t npage){
    return 0;
}

void IndexDiskV::trigger_local_merge_flag() const{
    for (auto& local_ptr : this->diskVectorHolders_local) {
        if (local_ptr) {
            local_ptr->need_merge.store(true, std::memory_order_relaxed);
        }
    }
}


void IndexDiskV::set_cache_strategy(CACHE_UPDATE_STRATEGY strategy, size_t batch_size, bool mode_1){
    
    if(strategy == NO_UPDATE){

    }else if(strategy == IMMEDIATELY_UPDATE){
        this->runtime_shard_update = true;
        this->batch_shard_update= false;
        this->runtime_cache = false;
        this->batch_shard_query_size= 0;
    }else if(strategy ==  SINGLE_THREAD_UPDATE){
        this->runtime_shard_update = true;
        this->batch_shard_update= true;
        this->runtime_cache = true;
        this->batch_shard_query_size= batch_size;
    }else if(strategy ==  GLOBAL_CONTROL_UPDATE){
        this->runtime_shard_update = false;
        this->batch_shard_update= false;
        this->runtime_cache = true;
        this->global_shard_update = true ;
        this->batch_shard_query_size= batch_size;
    }else if(strategy ==  SINGLE_THREAD_LOCAL_BUFFER_UPDATE){
        this->runtime_cache = true;
        this->cache_on_buffer = true;
        this->cache_buffer_size = batch_size;
        this->cache_buffer_remove_dup = mode_1;
    }else{
        FAISS_THROW_MSG("The strategy is not supported, change it.");
    }

}


namespace{
    void warm_invlist(
        std::string select_lists_path,
        std::vector<size_t>& sorted_idx,
        faiss::Aligned_Invlist_Info *aligned_inv_info,
        faiss::ClusteredArrayInvertedLists* c_invlists)
    {
        std::ifstream in_file(select_lists_path, std::ios::binary);
        if (!in_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + select_lists_path);
        }

        for (size_t idx : sorted_idx) {
            // Get cluster info
            const faiss::Aligned_Invlist_Info& inv_info = aligned_inv_info[idx];

            size_t ids_size = inv_info.list_size * sizeof(faiss::idx_t);
            size_t codes_size = inv_info.list_size * c_invlists->code_size;

            c_invlists->ids[idx].resize(inv_info.list_size);
            c_invlists->codes[idx].resize(codes_size);
            size_t data_offset = inv_info.page_start * PAGE_SIZE;
            in_file.seekg(data_offset, std::ios::beg);

            in_file.read(reinterpret_cast<char*>(c_invlists->ids[idx].data()), ids_size);
            in_file.read(reinterpret_cast<char*>(c_invlists->codes[idx].data()), codes_size);
        }

    }
}


int IndexDiskV::warmUpIndexMetadata(size_t n, float* x, size_t w_nprobe, size_t warm_list){
    this->cached_list_info = new bool[nlist];

    std::vector<idx_t> idx(n * w_nprobe);
    std::vector<float> coarse_dis(n * w_nprobe);
    std::vector<size_t> sorted_idx;
    // Search to get the nearest cluster IDs
    quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);

    sort_coarse(idx, sorted_idx, this->len, warm_list, static_cast<size_t>(-1));

    for(int i = 0; i < nlist; i++){
        this->cached_list_info[i] = false;
    }

    for(int i = 0; i < sorted_idx.size(); i++){
        this->cached_list_info[sorted_idx[i]] = true;
    }

    warm_invlist(disk_path, sorted_idx, this->aligned_inv_info, dynamic_cast<ClusteredArrayInvertedLists*>(this->invlists));

    return 0;
}



void IndexDiskV::warmUpAllIndexMetadata(){

    this->cached_list_info = new bool[nlist];

    std::vector<size_t> sorted_idx(this->nlist); // size should be n+1 to have indices from 0 to n
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);

        for(int i = 0; i < this->nlist; i++){
        this->cached_list_info[i] = true;
    }

    warm_invlist(disk_path, sorted_idx, this->aligned_inv_info, dynamic_cast<ClusteredArrayInvertedLists*>(this->invlists));

}

void IndexDiskV::initializeVectorCollector(int n_threads){
    vectorCollectors.resize(n_threads);
    for(int i = 0; i < n_threads; i++){
        vectorCollectors[i].reset(new VectorCollector(nlist));
    }
}

void IndexDiskV::initializeLocalVectorCaches(int n_threads, size_t capacity){
    this->diskVectorHolders_local.resize(n_threads);
    for(int i = 0; i < n_threads; i++){
        if(!cache_on_buffer)
            diskVectorHolders_local[i].reset(new DiskVectorHolder_local(capacity));
        else
            diskVectorHolders_local[i].reset(new DiskVectorHolder_local_buffer(capacity,get_code_size(),cache_buffer_remove_dup));
    }
}

void IndexDiskV::releaseLocalVectorCaches(int n_threads){

}


// Merge
std::vector<std::vector<VectorPair>> IndexDiskV::finalizeVectorCollector(int n_threads){
        std::vector<std::vector<VectorPair>> merged_result(nlist);
#pragma omp parallel for
        for (size_t list_no = 0; list_no < nlist; list_no++) {
            const idx_t* ids = this->invlists->get_ids(list_no);

            std::unordered_map<size_t, size_t> freq_map;
            for (int t = 0; t < n_threads; t++) {
                const auto &vec_data = vectorCollectors[t]->vector_collector[list_no];
                for (size_t vecid : vec_data) {
                    freq_map[vecid] += 1;
                }
            }

            std::vector<VectorPair> list_result;
            list_result.reserve(freq_map.size());
            for (const auto &it : freq_map) {
                VectorPair vp;
                vp.vecid = it.first;
                vp.freq  = it.second;
                vp.id = ids[vp.vecid]; 
                list_result.push_back(vp);
            }
            merged_result[list_no] = std::move(list_result);
        }

        for (int t = 0; t < n_threads; t++) {
            if (vectorCollectors[t]) {
                for (auto& list_vec : vectorCollectors[t]->vector_collector) {
                    list_vec.clear(); 
                    list_vec.shrink_to_fit();
                }
                vectorCollectors[t]->vector_collector.clear(); 
                vectorCollectors[t]->vector_collector.shrink_to_fit();
            }
        }

        vectorCollectors.clear();
        vectorCollectors.shrink_to_fit(); 

        return merged_result;
}




void IndexDiskV::initializeDiskIO(int n_threads){
    diskIOprocessors.resize(n_threads);
    for (int i = 0; i < n_threads; i++) {
        diskIOprocessors[i].reset(get_DiskIOSearchProcessor());
        diskIOprocessors[i]->initial();
    }
}

void IndexDiskV::shutdownDiskIO(int n_threads){
    diskIOprocessors.clear();
}

void IndexDiskV::multi_ivf_modify_id(){
    
    if(use_multi_ivf){
        std::cout << "Modifying ids in index " << order << " with partition_num " << partition_num<< "  ....\n";
        for(int i = 0; i < nlist; i++){

            size_t list_size = invlists->list_size(i);

            const uint8_t* origin_codes = invlists->get_codes(i);
            const idx_t* origin_ids = invlists->get_ids(i);
            std::vector<idx_t> modified_ids(list_size);

            for(int j = 0; j < list_size; j++){
                modified_ids[j] = origin_ids[j]*partition_num + order;
            }

            invlists->update_entries(i, 0, list_size, modified_ids.data(), origin_codes);
        }
    }
    else{
        FAISS_THROW_MSG("Don't modify ID in single index");
        
    }

    
}

void IndexDiskV::search(
        idx_t n,
        const float* x,
        idx_t k_r,
        float* distances_result,
        idx_t* labels_result,
        const SearchParameters* params_in ) const {
    FAISS_THROW_IF_NOT(k_r > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // : make a new distances and labels to contain replica*k results
    idx_t k = k_r * this->assign_replicas;
    std::unique_ptr<idx_t[]> del1(new idx_t[n * k]);
    std::unique_ptr<float[]> del2(new float[n * k]);
    idx_t* labels = del1.get();
    float* distances = del2.get();

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        auto time_start = std::chrono::high_resolution_clock::now();

        double t0 = getmillisecs();

        if(centroid_index != nullptr && nprobe > 10){
            centroid_index->hnsw.efSearch = std::min(nprobe,(size_t)200);
            centroid_index->search(n,x,nprobe,coarse_dis.get(),idx.get());
        }
        else{
            quantizer->search(
                    n,
                    x,
                    nprobe,
                    coarse_dis.get(),
                    idx.get(),
                    params ? params->quantizer_params : nullptr);
        }

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        auto time_end = std::chrono::high_resolution_clock::now();
        indexDiskV_stats.coarse_elapsed += time_end - time_start;

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    int nslice = omp_get_max_threads();
#pragma omp parallel for
    for (int slice = 0; slice < nslice; slice++) {
        idx_t i0 = n * slice / nslice;
        idx_t i1 = n * (slice + 1) / nslice;
        idx_t n_slice = i1 - i0;
        const float* x_i = x + i0 * d;
        float* dis_i = distances + i0 * k;
        idx_t* lab_i = labels + i0 * k;

        if(use_search_threshold)
        {
            int thread_id = omp_get_thread_num();
#pragma omp critical
            {
                q_indicator[thread_id] = i0;
            }
        }
        IndexIVFStats local_stats;
        sub_search_func(n_slice, x_i, dis_i, lab_i, &local_stats);
    }

    if(this->vector_cache_setting_mode || this->vector_cache_dp_setting_mode){
        return;
    }

    auto time_start = std::chrono::high_resolution_clock::now();      // time begin

    for(idx_t ii = 0; ii < n;ii++){
        idx_t begin_r = ii*k_r;
        idx_t begin = ii*k;
        idx_t limit = 0;

        for(idx_t jj = 0; jj < k; jj++){
            if(jj == 0){
                distances_result[begin_r] = distances[begin];
                labels_result[begin_r] = labels[begin];
                limit++;
            }
            else{
                if(labels[begin+jj] != labels[begin+jj-1]){
                    distances_result[begin_r+limit] = distances[begin+jj];
                    labels_result[begin_r+limit] = labels[begin+jj];
                    limit++;
                }
                if(limit>=k_r){
                    break;
                }

            }
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now(); 
    indexDiskV_stats.rerank_elapsed += time_end - time_start;
}


namespace{

struct DiskResultHandler{
    virtual void add(size_t q, float d0, idx_t id) = 0;
    virtual void end() = 0;
};

template<class C>
struct DiskHeapHandler : DiskResultHandler {
    float* dis;
    idx_t* ids;

    size_t k; // number of results to keep
    size_t nq;
    size_t nup;

    DiskHeapHandler(size_t nq, size_t k, float* dis, idx_t* ids)
            : nq(nq),
              dis(dis),
              ids(ids),
              k(k),
              nup(0) {
        heap_heapify<C>(k * nq, dis, ids);
    }

    void add(size_t q, float d0, idx_t id) override {
        float* current_dis = dis + q*k;
        idx_t* current_ids = ids + q*k;
        if (C::cmp(current_dis[0], d0)) {
            heap_replace_top<C>(k, current_dis, current_ids, d0, id);
            nup++;
        }

    }

    void end() override{
        for(int i = 0; i < nq; i++){
            heap_reorder<C>(k, dis + i*k, ids + i*k);
        }
    }

};


DiskResultHandler* get_result_handler(idx_t n, idx_t k, float* distances, idx_t* labels, MetricType metricType = METRIC_L2){
    if(metricType == METRIC_L2){
        return new DiskHeapHandler<CMax<float, idx_t>>(n, k, distances, labels);
    }else if(metricType == METRIC_INNER_PRODUCT){
        return new DiskHeapHandler<CMin<float, idx_t>>(n, k, distances, labels);
    }else{
        FAISS_THROW_MSG("Not support now!");
    }
}


struct UncachedList{
    size_t q;
    std::vector<faiss::idx_t> list_pos; 
};

}
// decode PQ begin
namespace{

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0
struct QueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const IndexIVFPQ& ivfpq;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQ for easier access
    int d;
    const ProductQuantizer& pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;
    int polysemous_ht;

    // pre-allocated data buffers
    float *sim_table, *sim_table_2;
    float *residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float*> sim_table_ptrs;

    explicit QueryTables(
            const IndexIVFPQ& ivfpq,
            const IVFSearchParameters* params)
            : ivfpq(ivfpq),
              d(ivfpq.d),
              pq(ivfpq.pq),
              metric_type(ivfpq.metric_type),
              by_residual(ivfpq.by_residual),
              use_precomputed_table(ivfpq.use_precomputed_table) {
        mem.resize(pq.ksub * pq.M * 2 + d * 2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        polysemous_ht = ivfpq.polysemous_ht;
        if (auto ivfpq_params =
                    dynamic_cast<const IVFPQSearchParameters*>(params)) {
            polysemous_ht = ivfpq_params->polysemous_ht;
        }
        if (polysemous_ht != 0) {
            q_code.resize(pq.code_size);
        }
        init_list_cycles = 0;
        sim_table_ptrs.resize(pq.M);
    }

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float* qi;

    // query-specific initialization
    void init_query(const float* qi) {
        this->qi = qi;
        if (metric_type == METRIC_INNER_PRODUCT)
            init_query_IP();
        else
            init_query_L2();
        if (!by_residual && polysemous_ht != 0)
            pq.compute_code(qi, q_code.data());
    }

    void init_query_IP() {
        // precompute some tables specific to the query qi
        pq.compute_inner_prod_table(qi, sim_table);
    }

    void init_query_L2() {
        if (!by_residual) {
            pq.compute_distance_table(qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table(qi, sim_table_2);
        }
    }

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    idx_t key;
    float coarse_dis;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                dis0 = precompute_list_tables_IP();
            else
                dis0 = precompute_list_tables_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    float precompute_list_table_pointers() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                FAISS_THROW_MSG("not implemented");
            else
                dis0 = precompute_list_table_pointers_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    /*****************************************************
     * compute tables for inner prod
     *****************************************************/

    float precompute_list_tables_IP() {
        // prepare the sim_table that will be used for accumulation
        // and dis0, the initial value
        ivfpq.quantizer->reconstruct(key, decoded_vec);
        // decoded_vec = centroid
        float dis0 = fvec_inner_product(qi, decoded_vec, d);

        if (polysemous_ht) {
            for (int i = 0; i < d; i++) {
                residual_vec[i] = qi[i] - decoded_vec[i];
            }
            pq.compute_code(residual_vec, q_code.data());
        }
        return dis0;
    }

    /*****************************************************
     * compute tables for L2 distance
     *****************************************************/

    float precompute_list_tables_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 0 || use_precomputed_table == -1) {
            ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);

            if (polysemous_ht != 0) {
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd(
                    pq.M * pq.ksub,
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M,
                    -2.0,
                    sim_table_2,
                    sim_table);

            if (polysemous_ht != 0) {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            const float* qtab = sim_table_2; // query-specific table
            float* ltab = sim_table;         // (output) list-specific table

            long k = key;
            for (int cm = 0; cm < cpq.M; cm++) {
                // compute PQ index
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                // get corresponding table
                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                if (polysemous_ht == 0) {
                    // sum up with query-specific table
                    fvec_madd(Mf * pq.ksub, pc, -2.0, qtab, ltab);
                    ltab += Mf * pq.ksub;
                    qtab += Mf * pq.ksub;
                } else {
                    for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
                        q_code[m] = fvec_madd_and_argmin(
                                pq.ksub, pc, -2, qtab, ltab);
                        pc += pq.ksub;
                        ltab += pq.ksub;
                        qtab += pq.ksub;
                    }
                }
            }
        }

        return dis0;
    }

    float precompute_list_table_pointers_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            const float* s =
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs[m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float* pc = ivfpq.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                for (int m = m0; m < m0 + Mf; m++) {
                    sim_table_ptrs[m] = pc;
                    pc += pq.ksub;
                }
                m0 += Mf;
            }
        } else {
            FAISS_THROW_MSG("need precomputed tables");
        }

        if (polysemous_ht) {
            FAISS_THROW_MSG("not implemented");
            // Not clear that it makes sense to implemente this,
            // because it costs M * ksub, which is what we wanted to
            // avoid with the tables pointers.
        }

        return dis0;
    }
};


template <typename IDType, class PQDecoder>
struct IVFPQScannerT : QueryTables {
    const uint8_t* list_codes;
    const IDType* list_ids;
    size_t list_size;

    IVFPQScannerT(const IndexIVFPQ& ivfpq, const IVFSearchParameters* params)
            : QueryTables(ivfpq, params) {
    }

    float dis0;

    void init_list(idx_t list_no, float coarse_dis, int mode) {
        this->key = list_no;
        this->coarse_dis = coarse_dis;

        if (mode == 2) {
            dis0 = precompute_list_tables();
            //std::cout << "dis0:" << dis0 << std::endl;
        } else if (mode == 1) {
            dis0 = precompute_list_table_pointers();
        }
    }

    size_t scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            float* local_dis,
            idx_t* local_ids) const {
        int counter = 0;
        size_t operations = 0;

        //std::cout << " pq.M:" << pq.M << "  pq.nbits:" << pq.nbits << std::endl;
        //assert(codes != nullptr);
        //assert(local_dis != nullptr);
        //assert(local_ids != nullptr);
        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            //std::cout << "ncode:"<< ncode <<"  j:" << j << std::endl;
            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                *(local_dis++) = dis0 + distance_0;
                *(local_ids++) = saved_j[0];
                *(local_dis++) = dis0 + distance_1;
                *(local_ids++) = saved_j[1];
                *(local_dis++) = dis0 + distance_2;
                *(local_ids++) = saved_j[2];
                *(local_dis++) = dis0 + distance_3;
                *(local_ids++) = saved_j[3];

                counter = 0;
                operations+=4;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[0];
            operations++;
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[1];
            operations++;
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[2];
            operations++;
        }

        return operations;
    }

};

struct BaseDecoder{

    idx_t list_no;
    bool store_pairs;
    virtual void set_query(const float* query) = 0;
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;
    virtual float distance_to_code(const uint8_t* code) const = 0;
    virtual size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* decoded_dis,
            idx_t* decoded_ids) const = 0;

};

template <class C, class PQDecoder>
struct IVFPQDecoder : IVFPQScannerT<idx_t, PQDecoder>, BaseDecoder {

    int precompute_mode;
    const IDSelector* sel;

    IVFPQDecoder(
            const IndexIVFPQ& ivfpq,
            bool store_pairs,
            int precompute_mode,
            const IDSelector* sel)
            : IVFPQScannerT<idx_t, PQDecoder>(ivfpq, nullptr),
              precompute_mode(precompute_mode),
              sel(sel) {
        this->store_pairs = store_pairs;
    }

    void set_query(const float* query) override {
        this->init_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->init_list(list_no, coarse_dis, precompute_mode);
    }

    float distance_to_code(const uint8_t* code) const override {
        assert(precompute_mode == 2);
        float dis = this->dis0 +
                distance_single_code<PQDecoder>(
                            this->pq.M, this->pq.nbits, this->sim_table, code);
        return dis;
    }

    size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* decoded_dis,
            idx_t* decoded_ids) const override{

        //std::cout << " scanning codes" <<std::endl;
        return this->scan_list_with_table(ncode, codes, decoded_dis, decoded_ids);
    }

    // void scan_codes_range(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         const idx_t* ids,
    //         float radius,
    //         RangeQueryResult& rres) const override {

    // }
};

template<class C, class PQDecoder>
BaseDecoder* get_pqdecoder1(const IndexIVFPQ& index){
    return new IVFPQDecoder<C, PQDecoder>(index, false, 2, nullptr);
}

template<class C>
BaseDecoder* get_pqdecoder2(const IndexIVFPQ& index){
    if (index.pq.nbits == 8) {
        return get_pqdecoder1<C, PQDecoder8>(
                index);
    } else if (index.pq.nbits == 16) {
        return get_pqdecoder1<C, PQDecoder16>(
                index);
    } else {
        return get_pqdecoder1<C, PQDecoderGeneric>(
                index);
    }
}

BaseDecoder* get_pqdecoder(const IndexIVFPQ& index, MetricType metricType){
    if(metricType == METRIC_L2){
        return get_pqdecoder2<CMax<float, idx_t>>(index);
    }else if(metricType == METRIC_INNER_PRODUCT){
        return get_pqdecoder2<CMin<float, idx_t>>(index);
    }else{
        FAISS_THROW_MSG("Do not support !!");
    }

}

// // read fully
struct ListSegment{
    // TODO: collect merged-read info such as maps, list ids, list distances, etc.
};

// read partially
struct PageSegment {
    idx_t list_no;
    int start_page;
    int page_count;
    int* in_buffer_offsets;    // Use pointers instead of std::vector
    size_t* in_buffer_ids;     // Use pointers instead of std::vector
    size_t length;             // Length parameter

    PageSegment()
        : in_buffer_offsets(nullptr), in_buffer_ids(nullptr), length(0) {}

    PageSegment(int start, int count, idx_t list_no, const int* offsets, const size_t* ids, size_t len)
        : start_page(start), page_count(count), list_no(list_no), length(len) {

        // Allocate memory and perform a deep copy
        in_buffer_offsets = (int*)malloc(length * sizeof(int));
        in_buffer_ids = (size_t*)malloc(length * sizeof(size_t));

        if (in_buffer_offsets && in_buffer_ids) {
            std::memcpy(in_buffer_offsets, offsets, length * sizeof(int));
            std::memcpy(in_buffer_ids, ids, length * sizeof(size_t));
        } else {
            // Handle allocation failure
            free(in_buffer_offsets);
            free(in_buffer_ids);
            in_buffer_offsets = nullptr;
            in_buffer_ids = nullptr;
            length = 0;
        }
    }

    // Destructor releases dynamically allocated memory
    ~PageSegment() {
        if(in_buffer_offsets != nullptr){
            free(in_buffer_offsets);
            free(in_buffer_ids);
        }
    }

    PageSegment(const PageSegment& other)
        : start_page(other.start_page), page_count(other.page_count), list_no(other.list_no), length(other.length) {

        in_buffer_offsets = (int*)malloc(length * sizeof(int));
        in_buffer_ids = (size_t*)malloc(length * sizeof(size_t));

        if (in_buffer_offsets && in_buffer_ids) {
            std::memcpy(in_buffer_offsets, other.in_buffer_offsets, length * sizeof(int));
            std::memcpy(in_buffer_ids, other.in_buffer_ids, length * sizeof(size_t));
        } else {
            free(in_buffer_offsets);
            free(in_buffer_ids);
            in_buffer_offsets = nullptr;
            in_buffer_ids = nullptr;
            length = 0;
        }
    }
};

} // anonymous namespace
// decode PQ end

// sort func begin
namespace{
    template <typename T1, typename T2>
    std::vector<size_t> sort_two_array(const T1* first_begin, const T2* second_begin, T1* result_first, T2* result_second, size_t num) {
        std::vector<size_t> indices(num);
        for (size_t i = 0; i < num; i++) {
            indices[i] = i;
        }

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return first_begin[a] < first_begin[b];
        });

        std::vector<T1> sorted_first(num);
        std::vector<T2> sorted_second(num);

        for (size_t i = 0; i < num; i++) {
            sorted_first[i] = first_begin[indices[i]];
            sorted_second[i] = second_begin[indices[i]];
        }

        std::copy(sorted_first.begin(), sorted_first.end(), result_first);
        std::copy(sorted_second.begin(), sorted_second.end(), result_second);
        return indices;
    }

    void decode_pq_lists(BaseDecoder* pqdecoder,
                     const float* query,
                     const idx_t* list_ids,
                     const float* list_dis,
                     size_t probe_batch,             
                     InvertedLists* invlists,
                     std::vector<std::vector<float>>& pq_distances,
                     std::vector<std::vector<idx_t>>& pq_ids,
                     const bool* cached_list_info = nullptr)
    {
        if(query)
            pqdecoder->set_query(query);

        for (size_t i = 0; i < probe_batch; i++) {
            idx_t list_no = list_ids[i];
            float list_distance = list_dis[i];

            size_t list_size = invlists->list_size(list_no);

            pq_distances[i].resize(list_size);
            pq_ids[i].resize(list_size);

#ifdef CACHE_MODE
            if(cached_list_info != nullptr){
                if(!cached_list_info[list_no])
                    continue;                  
            }
#endif
            pqdecoder->set_list(list_no, list_distance); 
            pqdecoder->scan_codes(list_size,
                                invlists->get_codes(list_no),
                                invlists->get_ids(list_no),
                                pq_distances[i].data(),
                                pq_ids[i].data());

        }
    }

    void decode_pq_lists(BaseDecoder* pqdecoder,
                     const float* query,
                     const idx_t* list_ids,
                     const float* list_dis,
                     size_t pqed_list,
                     size_t probe_batch,
                     InvertedLists* invlists,
                     std::vector<std::vector<float>>& pq_distances,
                     std::vector<std::vector<idx_t>>& pq_ids,
                     const bool* cached_list_info = nullptr)
    {
        if(query)
            pqdecoder->set_query(query);

        for (size_t i = 0; i < probe_batch; i++) {
            //std::cout << " Decoding list: " << (pqed_list + i) << std::endl;
            idx_t list_no = list_ids[pqed_list + i];
            float list_distance = list_dis[pqed_list + i];

            size_t list_size = invlists->list_size(list_no);

            pq_distances[pqed_list + i].resize(list_size);
            pq_ids[pqed_list + i].resize(list_size);

#ifdef CACHE_MODE
            if(cached_list_info != nullptr){
                if(!cached_list_info[list_no])
                    continue;
            }
#endif
            pqdecoder->set_list(list_no, list_distance);
            pqdecoder->scan_codes(list_size,
                                invlists->get_codes(list_no),
                                invlists->get_ids(list_no),
                                pq_distances[pqed_list + i].data(),
                                pq_ids[pqed_list + i].data());

        }
    }

    struct DiskInvlist{

        size_t list_size;
        size_t code_size;
        size_t list_no;

        std::vector<faiss::idx_t> disk_ids;
        std::vector<uint8_t> disk_codes;
        std::vector<size_t> disk_map;

        DiskInvlist() : list_size(0), code_size(0), list_no(0) {}
        DiskInvlist(void* disk_buffer, size_t list_size, size_t code_size, size_t list_no = 0)
            : list_size(list_size), code_size(code_size), list_no(list_no) {
            set(disk_buffer, list_size, code_size, list_no);
        }

        void set(void* disk_buffer, size_t list_size, size_t code_size, size_t list_no = 0) {
            this->list_size = list_size;
            this->code_size = code_size;
            this->list_no = list_no;

            size_t ids_size = list_size * sizeof(faiss::idx_t);
            size_t codes_size = list_size * code_size;
            size_t map_size = list_size * sizeof(size_t);

            char* buffer = static_cast<char*>(disk_buffer);
            disk_ids.assign(reinterpret_cast<faiss::idx_t*>(buffer),
                            reinterpret_cast<faiss::idx_t*>(buffer + ids_size));
            buffer += ids_size;

            disk_codes.assign(reinterpret_cast<uint8_t*>(buffer),
                            reinterpret_cast<uint8_t*>(buffer + codes_size));
        }

        size_t get_size(){
            return disk_ids.size();
        }

        faiss::idx_t* get_ids() {
            return disk_ids.data();
        }

        uint8_t* get_codes() {
            return disk_codes.data();
        }

        size_t* get_map() {
            return disk_map.data();
        }

    };




    void decode_pq_list(
        BaseDecoder* pqdecoder,
        const float* query,
        idx_t list_no,
        float list_distance,
        size_t list_size,
        DiskInvlist* disk_invlist,
        std::vector<float>& pq_distance,
        std::vector<idx_t>& pq_ids){

        if(query)
            pqdecoder->set_query(query);   // when decode several discrete lists, we can set query out of this function

        pq_distance.resize(list_size);
        pq_ids.resize(list_size);


        pqdecoder->set_list(list_no, list_distance);
        pqdecoder->scan_codes(list_size,
                            disk_invlist->get_codes(),
                            disk_invlist->get_ids(),
                            pq_distance.data(),
                            pq_ids.data());


    }

}


// merge pages
namespace{

void merge_pages_transpage(
    std::vector<PageSegment>& merged_segments,
    Page_to_Search* ptr_page_to_search,
    int* ptr_vector_to_submit,
    int* ptr_vector_to_search,
    size_t* vec_page_proj,

    size_t vectors_num,
    int num_page_to_search,
    const int per_page_element,
    const int per_page_vector,
    const size_t d,
    const idx_t list_no,
    const size_t max_pages
    )
{
    int max_vectors_per_request = num_page_to_search * (per_page_vector+1);  

    int current_page = ptr_page_to_search[0].first; 
    int record_begin_page = current_page;
    int page_count = 1;

    int in_buffer_offsets[max_vectors_per_request];
    size_t in_buffer_ids[max_vectors_per_request];
    size_t offset_count = 0;

    int vec_rank = 0;
    int page_rank = 0;

    while(vec_rank < vectors_num){
        while(vec_rank < vectors_num && ptr_page_to_search[vec_rank].first == current_page){
            int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;
            in_buffer_offsets[offset_count] = inbuffer;
            in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
            page_count += ptr_page_to_search[vec_rank].last - ptr_page_to_search[vec_rank].first;
            current_page = ptr_page_to_search[vec_rank].last;
            offset_count++;
            vec_rank++;
        }
        if(vec_rank < vectors_num){
            int page_gap = ptr_page_to_search[vec_rank].first - current_page;

            if(page_gap < max_pages){
                page_count += (ptr_page_to_search[vec_rank].first - ptr_page_to_search[vec_rank-1].last);
                current_page = ptr_page_to_search[vec_rank].first;
            }else{
                merged_segments.push_back(PageSegment(record_begin_page, page_count, list_no,
                    in_buffer_offsets, in_buffer_ids, offset_count));
                record_begin_page = ptr_page_to_search[vec_rank].first;
                current_page = ptr_page_to_search[vec_rank].first;
                page_count = 1;
                offset_count = 0;
            }
        }else{
            merged_segments.push_back(PageSegment(record_begin_page, page_count, list_no,
                in_buffer_offsets, in_buffer_ids, offset_count)); 
            break;
        }   
    }
}

} // anonymous namespace

//Beam search
#define BLOCK_BEAM_SEARCH


#define FETCH_FULL 1.0
#define REDUCED_FULL_LIST
#define LOSSLESS_FULL_REDUCTION



#ifdef REDUCED_FULL_LIST
namespace{
    struct DiskReadResult {
        size_t begin_idx;  
        size_t end_idx;    
        size_t start_page; 
        size_t end_page;   
        size_t iobuffer_offset;
    };

    DiskReadResult optimizeDiskRead(size_t min, size_t max, size_t list_size, size_t per_page_element_num, size_t d) {
        
        size_t vector_size_bytes = d * PAGE_SIZE / per_page_element_num;
        size_t start_page = (min * vector_size_bytes) / PAGE_SIZE;
        size_t end_page = (max * vector_size_bytes) / PAGE_SIZE;

        size_t start_offset = start_page * PAGE_SIZE;

        size_t begin_idx;
        size_t iobuffer_offset = 0;
        if (start_offset % vector_size_bytes == 0) {
            begin_idx = start_offset / vector_size_bytes;
        } else {
            iobuffer_offset = vector_size_bytes - (start_offset % vector_size_bytes);
            begin_idx = (start_offset + iobuffer_offset) / vector_size_bytes;
        }
        size_t end_offset = std::min((end_page + 1) * PAGE_SIZE, list_size * vector_size_bytes);
        size_t end_idx = end_offset / vector_size_bytes;
        return {begin_idx, end_idx, start_page, end_page, iobuffer_offset};
    }

    struct MinMaxStats {
        size_t min = std::numeric_limits<size_t>::max();
        size_t min_2 = std::numeric_limits<size_t>::max();
        size_t max = 0;
        size_t max_2 = 0;
        bool skip = true;

        inline void not_skip(){
            skip = false;
        }

        inline bool skip_2(){
            return min_2 > max_2;
        }

        void update(size_t value) {
            if (value > max) {
                max_2 = max;  
                max = value;
            } else if (value > max_2 && value < max) {
                max_2 = value;
            }

            if (value < min) {
                min_2 = min;
                min = value;
            } else if (value < min_2 && value > min) {
                min_2 = value;
            }
        }

        size_t get_min2(){
            if(min_2 > 500000)
                return 0;
            else{
                if(max_2 < min_2)
                    return max_2; 
                else
                    return min_2;
            }
        }

        size_t get_max2(){
            if(min_2 > 500000)
                return 0;
            if(max_2 < min_2)
                return min_2; 
            else
                return max_2;
        }
    };


    void calculateMinMax(const float* pq_dis_c, size_t vectors_num, float estimator, size_t k, std::priority_queue<float>& queue,
                     MinMaxStats& stats) {
        for (size_t m = 0; m < vectors_num; m++) {
            float value = pq_dis_c[m];
            if (queue.size() < k) {
                queue.push(value);
            } 
            else if (value < queue.top()) {
                queue.pop();
                queue.push(value);
            }
        }
        for (size_t m = 0; m < vectors_num; m++) {
            if (pq_dis_c[m] < queue.top() * estimator) {
                stats.update(m);
                stats.not_skip();
            }
        }
    }

    void calculateMinMax_ip(const float* pq_dis_c, size_t vectors_num, float estimator, size_t k, std::priority_queue<float>& queue,
            MinMaxStats& stats) {
            for (size_t m = 0; m < vectors_num; m++) {
                float value = pq_dis_c[m];
                if (queue.size() < k) {
                    queue.push( ( -value ));
                } 
                else if (value < queue.top()) {
                    queue.pop();
                    queue.push( ( -value ));
                }
            }
            for (size_t m = 0; m < vectors_num; m++) {
                if ((-pq_dis_c[m]) < queue.top() * estimator) {
                    stats.update(m);
                    stats.not_skip();
                }
            }
        }


}
#endif

namespace{
    void dedp(
            size_t list_size, 
            tsl::robin_map<faiss::idx_t, float>& id_dis_map,
            const idx_t* dp_ids,
            std::vector<std::vector<char>>& duplicate_indicators,
            size_t current_dped_list,
            size_t dpi,
            size_t& dp_count
         ){
        for(size_t dpj = 0; dpj < list_size; dpj++){
            if (id_dis_map.find(dp_ids[dpj]) != id_dis_map.end()){
            duplicate_indicators[current_dped_list + dpi][dpj] = 0;
            dp_count++;
            }else{
                duplicate_indicators[current_dped_list + dpi][dpj] = 1;
            }
        }
    }
}

void IndexDiskV::search_o(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler
#ifdef CACHE_MODE
    ,UncachedLists& uncached_lists
#endif
    ) const{
        if (n == 0) {
            return;
        }
        size_t i0 = 0;
        int thread_id = omp_get_thread_num();
        if(use_search_threshold){    
            i0 = q_indicator[thread_id];
        }
        VectorCollector* vectorCollector = nullptr;
        if(vector_cache_setting_mode || vector_cache_dp_setting_mode){
            vectorCollector = vectorCollectors[thread_id].get();
        }

        DiskVectorHolder_local* diskVectorHolder_local = nullptr;
        
        if(runtime_cache){
            diskVectorHolder_local = diskVectorHolders_local[thread_id].get();
        }
        
        std::vector<std::vector<float>> pq_distances;
        std::vector<std::vector<idx_t>> pq_ids;
        pq_distances.resize(nprobe);
        pq_ids.resize(nprobe);

        // global variable
        size_t current_pqed_list = 0;
        size_t pq_stage = 0;   // 0 for full
                               // 1 for partial stage 1
                               // 2 for partial stage 2
                               // 3 for others

        size_t current_dped_list = 0;

        const size_t top_probes = std::min((size_t)nprobe, top);

        const bool* cached_list_info = this->cached_list_info;
        std::vector<idx_t> working_lists_ids(top_probes);
        std::vector<float> working_lists_dis(top_probes);

        int working_list = top_probes;
        int pq_done = 0;
        bool pq_switch = false;
        int pq_todo = 0;
        bool pq_cross_query = false;
        bool pq_cross_extra_decode = true;

        std::vector<std::vector<char>> duplicate_indicators;
        duplicate_indicators.resize(nprobe);

        auto recordVectorCollector = [&](idx_t list_no, const std::vector<int>& candidates) {
            if (!vectorCollector) {
                return;
            }
            auto& bucket = vectorCollector->vector_collector[list_no];
            bucket.insert(bucket.end(), candidates.begin(), candidates.end());
        };

        BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);
        bool update_run_time_cache = false;
        bool update_run_time_shard_cache = false;
        for(int cur_q = 0; cur_q < n; cur_q++){
            const float* current_query = x + cur_q*d;
            float* heap_sim = distances + cur_q*k;
            idx_t* heap_ids = labels + cur_q*k;
            current_pqed_list = top_probes;
            current_dped_list = top_probes;
            pq_stage = 0;

            auto time_start = std::chrono::high_resolution_clock::now();
            
            size_t pre_decode = top_probes;
            tsl::robin_map<idx_t, float> id_dis_map;
            id_dis_map.reserve(20000);
            tsl::robin_map<idx_t, float> id_dis_map_local;
            id_dis_map_local.reserve(10000);
            const bool is_l2_metric = (this->metric_type == METRIC_L2);

            auto computeDistanceFromRaw = [&](const uint8_t* raw) -> float {
                if (is_l2_metric) {
                    if (this->valueType == "uint8") {
                        return vec_L2sqr_h<uint8_t>(current_query, reinterpret_cast<const uint8_t*>(raw), d);
                    }
                    if (this->valueType == "int8") {
                        return vec_L2sqr_h<int8_t>(current_query, reinterpret_cast<const int8_t*>(raw), d);
                    }
                    return fvec_L2sqr(current_query, reinterpret_cast<const float*>(raw), d);
                }
                return fvec_inner_product(current_query, reinterpret_cast<const float*>(raw), d);
            };

            auto filterCachedVectors = [&](auto&& fetch_fn,
                                           std::vector<int>& candidates,
                                           std::vector<int>& submissions,
                                           const idx_t* ids,
                                           size_t list_size,
                                           auto&& on_hit) -> bool {
                if (candidates.empty()) {
                    return false;
                }

                std::vector<int> mask(list_size, 0);
                for (int candidate : candidates) {
                    mask[candidate] = 1;
                }

                for (int candidate : candidates) {
                    const uint8_t* raw = fetch_fn(candidate);
                    if (!raw) {
                        continue;
                    }
                    const float distance = computeDistanceFromRaw(raw);
                    on_hit(ids[candidate], distance);
                    mask[candidate] = 0;
                }

                candidates.clear();
                submissions.clear();
                for (size_t idx = 0; idx < list_size; ++idx) {
                    if (mask[idx]) {
                        const int as_int = static_cast<int>(idx);
                        candidates.push_back(as_int);
                        submissions.push_back(as_int);
                    }
                }

                return candidates.empty();
            };

            decode_pq_lists(pqdecoder, x+cur_q*d, keys + cur_q*nprobe, coarse_dis+cur_q*nprobe,
                            0, pre_decode, invlists, pq_distances, pq_ids, cached_list_info);

            auto time_end = std::chrono::high_resolution_clock::now();
            indexDiskV_stats.memory_1_elapsed += time_end - time_start;

            AsyncReadRequests_Full_PQDecode requests_full;
            requests_full.list_requests.reserve(top_probes);

            Aligned_Cluster_Info* acinfo;
            int pushed_lists = 0;
            int pre_pushed_lists = 0;
            int pushed_requests = 0;
#ifdef CACHE_MODE
            std::vector<idx_t> record_uncached;
            record_uncached.reserve(nprobe);
#endif
#ifdef REDUCED_FULL_LIST
            std::priority_queue<float> pq_queue;
#endif

            while(pushed_lists < top_probes){
                size_t list_no = keys[cur_q*nprobe + pushed_lists];
#ifdef CACHE_MODE
                if(!cached_list_info[list_no]){
                    record_uncached.push_back(pushed_lists++);
                    continue;
                }
#endif
                size_t list_size = invlists->list_size(list_no);
                const idx_t* ids = invlists->get_ids(list_no);
                acinfo = &aligned_cluster_info[list_no];
#ifdef CACHE_MODE
                auto cce_time_start = std::chrono::high_resolution_clock::now();
                int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                if(cluster_pos!= -1){
                    //std::cout << "using cached data, listno=" << list_no << "  cluster_pos="  << cluster_pos << std::endl;
                    float* pq_dis_c = pq_distances[pushed_lists].data();
                    const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                    float distance;

                    if(this->metric_type == METRIC_L2){
                        
                        for(int m = 0; m < list_size; m++){
                            if(pq_dis_c[m] < heap_sim[0]*this->estimate_factor){
                                if(this->valueType == "uint8")
                                    distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*m, d);
                                else if(this->valueType == "int8")
                                    distance = vec_L2sqr_h<int8_t>(current_query, ((int8_t*)buffer) +d*m, d);
                                else
                                    distance = fvec_L2sqr(current_query, (float*)buffer +d*m, d);
                                heap_handler->add(cur_q, distance, ids[m]);
                            }
                        }
                    }else{
                        for(int m = 0; m < list_size; m++){
                            if(pq_dis_c[m] > heap_sim[0]/this->estimate_factor){
                                // For now only handle float values
                                distance = fvec_inner_product(current_query, (float*)buffer +d*m, d);
                                heap_handler->add(cur_q, distance, ids[m]);
                            }
                        }
                    }

                    indexDiskV_stats.cached_list_access += 1;
                    indexDiskV_stats.cached_vector_access += list_size;
                    pushed_lists++;

                    //std::cout << "cache list end" << std::endl;
                    auto cce_time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                    continue;
                }
#endif
#ifndef REDUCED_FULL_LIST
                size_t reduced_begin_page = acinfo->page_start * PAGE_SIZE;
                size_t reduced_fetch_pages = (size_t)(acinfo->page_count * FETCH_FULL);
                if(reduced_fetch_pages == 0) reduced_fetch_pages++;
                size_t reduced_fetch_size = (size_t)(list_size * FETCH_FULL);
                size_t reduced_begin_idx = 0;
                size_t reduced_iobuffer_offset = 0;
#else
                
                MinMaxStats full_prune;
                if(this->metric_type == METRIC_L2)
                    calculateMinMax(pq_distances[pushed_lists].data(), list_size, estimate_factor_high_dim, k,  pq_queue, full_prune);
                else
                    calculateMinMax_ip(pq_distances[pushed_lists].data(), list_size, 1.0/estimate_factor_high_dim, k,  pq_queue, full_prune);
     
#ifdef LOSSLESS_FULL_REDUCTION
                if(full_prune.skip){
                    pushed_lists++;
                    continue;
                }
                DiskReadResult redret = optimizeDiskRead(full_prune.min, full_prune.max, list_size, diskIOprocessor->get_per_page_element(), d);
#else   
                if(full_prune.skip_2()){
                    pushed_lists++;
                    continue;
                }
                DiskReadResult redret = optimizeDiskRead(full_prune.get_min2(), full_prune.get_max2(), list_size, diskIOprocessor->get_per_page_element(), d);
#endif
                size_t reduced_begin_page = (acinfo->page_start + redret.start_page)  * PAGE_SIZE;
                size_t reduced_fetch_pages = redret.end_page - redret.start_page + 1;
                size_t reduced_begin_idx = redret.begin_idx;
                size_t reduced_fetch_size = redret.end_idx;
                size_t reduced_iobuffer_offset = redret.iobuffer_offset;

#endif
                if(reduced_fetch_pages == 0){
                    continue;
                }
                requests_full.list_requests.emplace_back(reduced_begin_page,
                                                    reduced_fetch_pages,
                                                    reduced_fetch_size,
                                                    reduced_begin_idx,
                                                    list_size,
                                                    reduced_iobuffer_offset,
                                                    nullptr,
                                                    ids,
                                                    pq_distances[pushed_lists].data());
                assert(list_size == pq_distances[pushed_lists].size());
                pushed_lists++;
                pushed_requests++;
                pre_pushed_lists++;
          
                indexDiskV_stats.searched_vector_full+=reduced_fetch_size;
                indexDiskV_stats.searched_page_full+=acinfo->page_count;
                faiss::indexDiskV_stats.searched_lists++;
            }

            indexDiskV_stats.requests_full+=pushed_requests;
            size_t decode_batch = std::max(top_probes*full_decode_volume, submit_per_round);
            decode_batch = std::min(decode_batch, nprobe - top_probes);


            requests_full.cal_callback = [&](AsyncRequest_Full* requested, void* buffer){
                auto time_start = std::chrono::high_resolution_clock::now();

                float distance;
                const idx_t* list_ids = requested->ids;
                const float* pq_dis_c = requested->pq_dis.data();

                int buffer_m = 0;
                uint8_t* right_buffer = ((uint8_t*)buffer) + requested->iobuffer_offset;
                if(this->metric_type == METRIC_L2){
                    for(int m = requested->begin_idx; m < requested->vectors_num; m++, buffer_m++){
                        if(pq_dis_c[m] < heap_sim[0]*this->estimate_factor)
                        {
                            if(this->valueType == "uint8")
                                distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)right_buffer) +d*buffer_m, d);
                            else if(this->valueType == "int8")
                                distance = vec_L2sqr_h<int8_t>(current_query, ((int8_t*)right_buffer) +d*buffer_m, d);    
                            else
                                distance = fvec_L2sqr(current_query, (float*)right_buffer +d*buffer_m, d);
                            heap_handler->add(cur_q, distance, list_ids[m]);

                            id_dis_map[list_ids[m]] = distance;
                        }else{
                            id_dis_map[list_ids[m]] = 0;
                        }
                    }
                }else{
                    for(int m = requested->begin_idx; m < requested->vectors_num; m++, buffer_m++){
                        if(pq_dis_c[m] > heap_sim[0]/this->estimate_factor){
                            distance = fvec_inner_product(current_query, (float*)buffer +d*m, d);
                            heap_handler->add(cur_q, distance, list_ids[m]);
                            id_dis_map[list_ids[m]] = distance;

                        }else{
                            id_dis_map[list_ids[m]] = 0;
                        }
                    }
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexDiskV_stats.rank_elapsed+=time_end - time_start;

            };
            requests_full.pq_callback = [&](){
                if(top_probes < nprobe){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    decode_pq_lists(pqdecoder, nullptr, keys + cur_q*nprobe, coarse_dis + cur_q*nprobe,
                                    current_pqed_list, decode_batch, invlists, pq_distances, pq_ids, cached_list_info);
                    current_pqed_list += decode_batch;
                    pq_stage = 1;
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.pq_elapsed+=time_end - time_start;
                    time_start = std::chrono::high_resolution_clock::now();
                    for(int dpi = 0; dpi < decode_batch; dpi++){
                        idx_t list_no = (keys + cur_q*nprobe)[current_dped_list + dpi];
                        size_t list_size = invlists->list_size(list_no);
                        duplicate_indicators[current_dped_list + dpi].resize(list_size, 1);
                    }
                    time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.full_duplicate_elapsed+=time_end - time_start;                    
                }
            };

            
            bool dp_increment = false;
            
            requests_full.dp_callback = [&](){
                auto time_start = std::chrono::high_resolution_clock::now();
                for(int dpi = 0; dpi < decode_batch; dpi++){
                    idx_t list_no = (keys + cur_q*nprobe)[current_dped_list + dpi];
                    size_t list_size = invlists->list_size(list_no);
                    const idx_t* dp_ids = invlists->get_ids(list_no);
                    size_t dp_count = 0;
                    if(!dp_increment){
                        dedp(list_size, id_dis_map, dp_ids, duplicate_indicators, current_dped_list, dpi, dp_count);
                    }else{
                        dedp(list_size, id_dis_map_local, dp_ids, duplicate_indicators, current_dped_list, dpi, dp_count);
                    }        
                 }     
                auto time_end = std::chrono::high_resolution_clock::now();
                indexDiskV_stats.full_duplicate_elapsed+=time_end - time_start;
            };

            requests_full.pq_callback();
            diskIOprocessor->disk_io_full_async_pq(requests_full);
            time_start = std::chrono::high_resolution_clock::now();
            diskIOprocessor->submit(pushed_requests);
            current_dped_list += decode_batch;
            time_end = std::chrono::high_resolution_clock::now();
            indexDiskV_stats.disk_full_elapsed+=time_end - time_start;

            

            working_list = top_probes;
            size_t actual_submit = 0;
            
            if(runtime_cache && local_merge_size){
                this->result_cached_queries[thread_id]++;
                if(this->result_cached_queries[thread_id] % local_merge_size == 0)
                    update_run_time_cache = true;
            }

            // shard single thread
            if(runtime_cache && batch_shard_update){
                this->result_cached_queries[thread_id]++;
                if(this->result_cached_queries[thread_id] % batch_shard_query_size == 0)
                    update_run_time_shard_cache = true;
            }

            // shard global update
            if(runtime_cache && global_shard_update){
                size_t new_count = global_query_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (new_count % batch_shard_query_size == 0) {
                    trigger_local_merge_flag();
                }
            }
            //std::cout <<"\n\n";

            while(working_list < nprobe){
                actual_submit = std::min((idx_t)submit_per_round, nprobe - (idx_t)working_list);
                if(pq_stage == 1){
                    pq_todo = std::min(submit_per_round * partial_one_decode_volume, nprobe - current_pqed_list);
                }else if(pq_stage == 2){
                    pq_todo = std::min(submit_per_round * partial_two_decode_volume, nprobe - current_pqed_list);
                }else if(pq_stage >= 3){
                    pq_todo = std::min(submit_per_round, nprobe - current_pqed_list);
                }
                pq_stage += 1;
                
                AsyncReadRequests_Partial_PQDecode request_p;
                for(int i = 0; i < actual_submit; i++){
                    idx_t list_no = keys[cur_q*nprobe + i + working_list];
#ifdef CACHE_MODE
                    if(!cached_list_info[list_no]){
                        record_uncached.push_back(i + working_list);
                        continue;
                    }
#endif
                    float cur_dis = coarse_dis[cur_q*nprobe + i + working_list];
                    float base_dis = coarse_dis[cur_q*nprobe];

                    if(this->centroid_index->metric_type == METRIC_L2){
                        if(cur_dis > base_dis * prune_factor){
                            break;
                        }
                    }else{
                        if(cur_dis < base_dis / prune_factor){
                            break;
                        }
                    }
                    
                    std::vector<int> vector_to_search;
                    std::vector<int> vector_to_submit;
                    size_t list_size = invlists->list_size(list_no);
             
                    auto time_start = std::chrono::high_resolution_clock::now();
                    int reserve_size = 0;
                    float* dis_line = pq_distances[working_list + i].data();
                    if((pq_distances[working_list + i].size() != list_size) ||
                        duplicate_indicators[working_list + i].size() != list_size){
                        std::cout << "working_list + i:" << working_list + i << "  list_size:" << list_size << std::endl;
                    }
                    float dynamic_estimate_factor_partial = estimate_factor_partial;
                    float dynamic_threshold = heap_sim[0] ;
                    
                    if(this->use_search_threshold && this->metric_type == METRIC_L2){
                        if(dynamic_threshold > this->search_threshold[cur_q+i0]){
                            dynamic_threshold = this->search_threshold[cur_q+i0];
                        }
                    }else if(this->use_search_threshold && this->metric_type == METRIC_INNER_PRODUCT){
                        if(dynamic_threshold < this->search_threshold[cur_q+i0]){
                            dynamic_threshold = this->search_threshold[cur_q+i0];
                        }
                    }
                    
                    size_t list_size_pruned = list_size;
                    
                    if(this->metric_type == METRIC_L2){
                        for(size_t j = 0; j < list_size_pruned; j++){
                            if (dis_line[j] < dynamic_threshold* dynamic_estimate_factor_partial){
                                if(duplicate_indicators[working_list + i][j]){
                                    reserve_size++;
                                }   
                            }
                        }
                    }else{
                        for(size_t j = 0; j < list_size_pruned; j++){
                            if (dis_line[j] > dynamic_threshold / dynamic_estimate_factor_partial){
                                if(duplicate_indicators[working_list + i][j]){
                                    reserve_size++;
                                }   
                            }
                        }
                    }
                    

                    if(reserve_size == 0){
                        continue;
                    }
                    faiss::indexDiskV_stats.searched_lists++;

                    vector_to_search.reserve(reserve_size + 10);
                    vector_to_submit.reserve(reserve_size + 10);
                    int reserve_next = 0;
                    if(this->metric_type == METRIC_L2){
                        for(size_t j = 0; j < list_size_pruned; j++){
                            if (dis_line[j] < dynamic_threshold* dynamic_estimate_factor_partial){
                                if(duplicate_indicators[working_list + i][j]){
                                    vector_to_search.push_back(j);
                                    vector_to_submit.push_back(j);
                                    reserve_next++;
                                }
                            }
                        }
                    }else{
                        for(size_t j = 0; j < list_size_pruned; j++){
                            if (dis_line[j] > dynamic_threshold / dynamic_estimate_factor_partial){
                                if(duplicate_indicators[working_list + i][j]){
                                    vector_to_search.push_back(j);
                                    vector_to_submit.push_back(j);
                                    reserve_next++;
                                }
                            }
                        }
                    }

                    if(vector_cache_setting_mode || vector_cache_dp_setting_mode){
                        recordVectorCollector(list_no, vector_to_search);
                    }


                    /* Cache vector processing begin*/
                    int cached_vectors_num = 0;
                    if(cached_vectors_num = diskVectorHolder.is_cached(list_no)){
                        const idx_t* tmp_ids =invlists->get_ids(list_no);
                        std::vector<int> candidates(list_size, 0);
                        for(size_t candidate : vector_to_search){
                            candidates[candidate] = 1;
                        }

                        const int* cached_vector_position = diskVectorHolder.get_cached_vector_position(list_no);
                        const uint8_t* cached_vector = diskVectorHolder.get_cache_data(list_no);
                        for(int c = 0; c < cached_vectors_num; c++){
                            if(candidates[cached_vector_position[c]] == 1){
                                float distance;
                                if(this->metric_type == METRIC_L2){
                                    if(this->valueType == "uint8")
                                        distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)cached_vector) +d*c, d);
                                    else if(this->valueType == "int8")
                                        distance = vec_L2sqr_h<int8_t>(current_query, ((int8_t*)cached_vector) +d*c, d);    
                                    else
                                        distance = fvec_L2sqr(current_query, (float*)cached_vector +d*c, d);

                                    heap_handler->add(cur_q, distance, tmp_ids[cached_vector_position[c]]);
                                }else{
                                    distance = fvec_inner_product(current_query, (float*)cached_vector +d*c, d);
                                    heap_handler->add(cur_q, distance, tmp_ids[cached_vector_position[c]]);
                                }
                                id_dis_map[tmp_ids[cached_vector_position[c]]] = distance;
                                candidates[cached_vector_position[c]] = 0;
                                faiss::indexDiskV_stats.cached_vectors_partial += 1;
                            }
                        }
                        int before_cache = vector_to_search.size();
                        vector_to_search.clear();
                        vector_to_submit.clear();
                        int after_cache = 0;
                        for(size_t j = 0; j < list_size; j++){
                            if(candidates[j] == 1){
                                vector_to_search.push_back(j);
                                vector_to_submit.push_back(j);
                                after_cache++;
                            }
                        }
                        if(after_cache == 0){
                            continue;
                        }

                    }
/**************************************************************************************************************/
                    if(diskVectorHolder_dp){
                            const idx_t* ids = this->invlists->get_ids(list_no);
                            std::vector<int> candidates(list_size, 0);
                            for(size_t candidate : vector_to_search){     
                                 candidates[candidate] = 1;
                            }

                            for(size_t candidate : vector_to_search){
                                const uint8_t* vecdata = diskVectorHolder_dp->get_cached_data(ids[candidate]);
                                if(vecdata){
                                    float distance;
                                    if(this->metric_type == METRIC_L2){
                                        if(this->valueType == "uint8")
                                            distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)vecdata), d);
                                        else if(this->valueType == "int8")
                                            distance = vec_L2sqr_h<int8_t>(current_query, ((int8_t*)vecdata) , d);    
                                        else
                                            distance = fvec_L2sqr(current_query, (float*)vecdata, d);

                                        heap_handler->add(cur_q, distance, ids[candidate]);
                                    }else{
                                        distance = fvec_inner_product(current_query, (float*)vecdata, d);
                                        heap_handler->add(cur_q, distance, ids[candidate]);
                                    }
                                    id_dis_map[ids[candidate]] = distance;

                                    candidates[candidate] = 0;
                                }
                            }
                            int before_cache = vector_to_search.size();
                            vector_to_search.clear();
                            vector_to_submit.clear();
                            int after_cache = 0;
                            for(size_t j = 0; j < list_size; j++){
                                if(candidates[j] == 1){
                                    vector_to_search.push_back(j);
                                    vector_to_submit.push_back(j);
                                    after_cache++;
                                }
                            }
                            if(after_cache == 0){
                                auto time_end = std::chrono::high_resolution_clock::now();
                                indexDiskV_stats.memory_2_elapsed+=time_end - time_start;
                                continue;
                            }
                    }
/**************************************************************************************************************/
                    if(runtime_cache && diskVectorHolder_local && !cache_on_buffer){
                        const idx_t* ids = this->invlists->get_ids(list_no);
                        std::shared_lock<std::shared_mutex> lock(diskVectorHolder_local->local_mutex);
                        auto runtime_fetch = [&](int candidate) {
                            return diskVectorHolder_local->get_cached_data(ids[candidate]);
                        };
                        auto on_runtime_hit = [&](idx_t vid, float distance) {
                            heap_handler->add(cur_q, distance, vid);
                            id_dis_map[vid] = distance;
                            id_dis_map_local[vid] = distance;
                            update_run_time_cache = true;
                        };

                        if (filterCachedVectors(runtime_fetch,
                                                vector_to_search,
                                                vector_to_submit,
                                                ids,
                                                list_size,
                                                on_runtime_hit)) {
                            auto time_end = std::chrono::high_resolution_clock::now();
                            indexDiskV_stats.memory_2_elapsed += time_end - time_start;
                            continue;
                        }
                    }
/**************************************************************************************************************/
/***************************************************************************************************************/
                    if(diskVectorHolder_shard){
                        const idx_t* ids = this->invlists->get_ids(list_no);
                        auto shard_fetch = [&](int candidate) {
                            return diskVectorHolder_shard->get(ids[candidate]);
                        };
                        auto on_shard_hit = [&](idx_t vid, float distance) {
                            heap_handler->add(cur_q, distance, vid);
                            id_dis_map[vid] = distance;
                        };

                        if (filterCachedVectors(shard_fetch,
                                                vector_to_search,
                                                vector_to_submit,
                                                ids,
                                                list_size,
                                                on_shard_hit)) {
                            auto time_end = std::chrono::high_resolution_clock::now();
                            indexDiskV_stats.memory_2_elapsed += time_end - time_start;
                            continue;
                        }
                    }
/**************************************************************************************************************/
                    /*Cache vector processing end*/
                    size_t vectors_num = vector_to_search.size();

#ifdef CACHE_MODE
                    auto cce_time_start = std::chrono::high_resolution_clock::now();
                    int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                    //std::cout << "cluster_pos:" << cluster_pos << std::endl;
                    if(cluster_pos!= -1){
                        std::cout << "cache!\n" ;
                        int* vec_pos = vector_to_search.data();
                        const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                        const idx_t* ids = invlists->get_ids(list_no);
                        float distance;
                        if(this->metric_type == METRIC_L2){
                            for(int m = 0; m < vectors_num; m++){
                                if(this->valueType == "uint8")
                                    distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*vec_pos[m], d);
                                else if(this->valueType == "int8")
                                    distance = vec_L2sqr_h<int8_t>(current_query, ((int8_t*)buffer) +d*vec_pos[m], d);
                                else
                                    distance = fvec_L2sqr(current_query, (float*)buffer +d*m, d);
                                heap_handler->add(cur_q, distance, ids[vec_pos[m]]);
                            }
                        }else{
                            for(int m = 0; m < vectors_num; m++){
                                distance = fvec_inner_product(current_query, (float*)buffer +d*m, d);
                                heap_handler->add(cur_q, distance, ids[vec_pos[m]]);
                            }
                        }
                        
                        indexDiskV_stats.cached_vector_access += vectors_num;
                        auto cce_time_end = std::chrono::high_resolution_clock::now();
                        indexDiskV_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                        continue;
                    }

#endif
                    if(vectors_num <= 0){
                        continue;
                    }

                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.memory_2_elapsed+=time_end - time_start;
                    time_start = std::chrono::high_resolution_clock::now();
                    std::vector<size_t> vec_page_proj(vectors_num);
                    std::vector<Page_to_Search> page_to_search(vectors_num);
                    int num_page_to_search = diskIOprocessor->process_page_transpage(vector_to_submit.data(), page_to_search.data(),
                                                            vec_page_proj.data(), vectors_num);

                    Aligned_Cluster_Info* cluster_info = &aligned_cluster_info[list_no];

                    const int per_page_element = diskIOprocessor->get_per_page_element();
                    const int per_page_vector = per_page_element/d;

                    bool not_aligned = false;
                    if(per_page_element%d != 0)
                        not_aligned = true;

                    std::vector<PageSegment> merged_segments;
                    int max_continous_pages = this->max_continous_pages;
                    merged_segments.reserve(num_page_to_search);

                    if (num_page_to_search > 0) {
                        merge_pages_transpage(merged_segments, page_to_search.data(), vector_to_submit.data(), vector_to_search.data(), vec_page_proj.data(),
                                    vectors_num, num_page_to_search, per_page_element, per_page_vector, d, list_no,max_continous_pages);
                    }else{
                        continue;
                    }

                    if(working_list < top){
                        faiss::indexDiskV_stats.searched_vector_full += vectors_num;
                        faiss::indexDiskV_stats.searched_page_full += num_page_to_search;
                        faiss::indexDiskV_stats.requests_full += merged_segments.size();
                    }
                    else{
                        faiss::indexDiskV_stats.searched_vector_partial += vectors_num;

                        for(int seg = 0; seg<merged_segments.size(); seg++){
                            faiss::indexDiskV_stats.searched_page_partial += merged_segments[seg].page_count;
                        }
                        faiss::indexDiskV_stats.requests_partial += merged_segments.size();
                    }

                    /*Async IO info*/
                    size_t global_start = cluster_info->page_start;
                    size_t prepare_size = request_p.list_requests.size();
                    request_p.list_requests.reserve(prepare_size + merged_segments.size());
                    for (size_t j = 0; j < merged_segments.size(); ++j) {
                        const auto& segment = merged_segments[j];

                        size_t page_num = segment.page_count;
                        size_t start_page = segment.start_page;
                        size_t offset = (global_start + segment.start_page) * PAGE_SIZE;
                        size_t iobuffer_offset = (d - (start_page*per_page_element) %d ) % d; //? sift ==0
                        size_t total_vector_num = (page_num * per_page_element - iobuffer_offset)/d;
                        size_t begin_idx = (start_page*per_page_element + d - 1)/d;
                        if(begin_idx + total_vector_num > list_size){
                            total_vector_num = list_size - begin_idx;
                        }
                        const size_t* map = nullptr;
                        const idx_t* ids = invlists->get_ids(segment.list_no);

                        int* in_buffer_begin = segment.in_buffer_offsets;
                        int* in_buffer_end = segment.in_buffer_offsets + segment.length;

                        size_t* in_ids_start = segment.in_buffer_ids;
                        size_t* in_ids_end = segment.in_buffer_ids + segment.length;

                        request_p.list_requests.emplace_back(page_num, total_vector_num, begin_idx, offset, iobuffer_offset, map, ids,
                        in_buffer_begin, in_buffer_end, in_ids_start, in_ids_end, segment.list_no);

                    }

                    time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.memory_3_elapsed+=time_end - time_start;
                }

                request_p.cal_callback = [&](AsyncRequest_Partial* requested, void* buffer){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    int* element_offsets = requested->in_buffer_offsets.data();
                    size_t* element_ids = requested->in_buffer_ids.data();

                    float distance;
                    const idx_t * list_ids = requested->ids;
                    std::vector<float> float_vector(d);
#ifndef BLOCK_BEAM_SEARCH
                    if(this->metric_type == METRIC_L2){
                        for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                            distance = fvec_L2sqr(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                            heap_handler->add(cur_q, distance, list_ids[element_ids[m]]);
                            id_dis_map[list_ids[element_ids[m]]] = distance;
                        }
                    }else{
                        for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                            distance = fvec_inner_product(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                            heap_handler->add(cur_q, distance, list_ids[element_ids[m]]);
                            id_dis_map[list_ids[element_ids[m]]] = distance;
                        }
                    }
                    
#else
                    if(this->metric_type == METRIC_L2){
                        for(int m = 0; m < requested->vectors_num; m++){
                            distance = fvec_L2sqr(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d + requested->iobuffer_offset), d);
                            heap_handler->add(cur_q, distance, list_ids[requested->begin_idx + m]);
                            id_dis_map[list_ids[requested->begin_idx + m]] = distance;
                        }
                    }else{
                        for(int m = 0; m < requested->vectors_num; m++){
                            distance = fvec_inner_product(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d + requested->iobuffer_offset), d);
                            heap_handler->add(cur_q, distance, list_ids[requested->begin_idx + m]);
                            id_dis_map[list_ids[requested->begin_idx + m]] = distance;
                        }
                    }
             
#endif
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.rank_elapsed+=time_end - time_start;
                    
                    time_start = std::chrono::high_resolution_clock::now();

                    size_t code_size = this->get_code_size();
                    size_t element_size = code_size/d;
                    // Insert here
                    if(runtime_cache && diskVectorHolder_local){
                        //std::cout << "Insert Local!" <<std::endl;
                        // BUG: this appears to misbehave
                        for(int m = 0; m < requested->in_buffer_offsets.size(); m++)
                        {
                            diskVectorHolder_local->insert_cache(list_ids[element_ids[m]], (uint8_t*)buffer + element_size * element_offsets[m], code_size);
                            //std::cout << "element_ids[m]: " <<element_ids[m] << "  element_offsets[m]: "<< 
                        }
                    }
                    if(runtime_shard_update && diskVectorHolder_shard ){
                        if(!batch_shard_update){
                            for(int m = 0; m < requested->in_buffer_offsets.size(); m++)
                            {
                                diskVectorHolder_shard->insert(list_ids[element_ids[m]], (uint8_t*)buffer + element_size * element_offsets[m]);
                                //std::cout << "element_ids[m]: " <<element_ids[m] << "  element_offsets[m]: "<< 
                            }
                        }
                        
                    }
                    
                    if(cache_on_buffer && diskVectorHolder_local && diskVectorHolder_local->need_merge.load(std::memory_order_relaxed)){
                        //std::cout << "buffer_cache.\n";
                        diskVectorHolder_shard->insert_batch_buffer(diskVectorHolder_local);
                    }
                    time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.cache_system_insert_elapsed += time_end - time_start;
                };

                request_p.pq_callback = [&](){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    if(pq_todo > 0){
                        decode_pq_lists(pqdecoder, nullptr, keys+cur_q*nprobe, coarse_dis+cur_q*nprobe,
                                        current_pqed_list, pq_todo, invlists, pq_distances, pq_ids, this->cached_list_info);
                        for(int dpi = 0; dpi < pq_todo; dpi++){
                            idx_t list_no = (keys + cur_q*nprobe)[current_dped_list + dpi];
                            size_t list_size = invlists->list_size(list_no);
                            duplicate_indicators[current_dped_list + dpi].resize(list_size, 1);
                        }
                        current_pqed_list += pq_todo;
                        indexDiskV_stats.pq_list_partial+=pq_todo;
                        pq_done += pq_todo;
                        //std::cout << "thread:" << thread_id <<"PQ decode finished\n";
                    }
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.pq_elapsed+=time_end - time_start;
                    

                };

                dp_increment = false;
                
                request_p.dp_callback = [&](){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    for(int dpi = 0; dpi < pq_todo; dpi++){
                        idx_t list_no = (keys + cur_q*nprobe)[current_dped_list + dpi];
                        size_t list_size = invlists->list_size(list_no);
                        const idx_t* dp_ids = invlists->get_ids(list_no);
                        size_t dp_count = 0;
                        if(!dp_increment){
                            dedp(list_size, id_dis_map, dp_ids, duplicate_indicators, current_dped_list, dpi, dp_count);
                        }
                    }
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexDiskV_stats.partial_duplicate_elapsed+=time_end - time_start;

                };

                 //std::cout << "Start Partial PQ\n";
                 //request_p.pq_callback();
                //std::cout << "Start Disk IO\n";
                diskIOprocessor->disk_io_partial_async_pq(request_p);
                auto time_start = std::chrono::high_resolution_clock::now();
                diskIOprocessor->submit();
                auto time_end = std::chrono::high_resolution_clock::now();
                indexDiskV_stats.disk_partial_elapsed+=time_end - time_start;

                

                current_dped_list += pq_todo;
                working_list+=actual_submit;

                if(update_run_time_shard_cache && diskVectorHolder_shard){
                    if(batch_shard_update && diskVectorHolder_local){
                        diskVectorHolder_shard->insert_batch(diskVectorHolder_local);
                    }
                    update_run_time_shard_cache = false;
                    
                }

                if (global_shard_update && diskVectorHolder_local && diskVectorHolder_local->need_merge.load(std::memory_order_relaxed)) {
                    diskVectorHolder_shard->insert_batch(diskVectorHolder_local);
                    diskVectorHolder_local->need_merge.store(false, std::memory_order_relaxed);
                }

                if(update_run_time_cache && diskVectorHolder_local){
                    if(!this->switch_result_cached_queries){
                        diskVectorHolder_dp->merge_from_local(diskVectorHolder_local);
                    }
                    else{
                        merge_cache_local_to_global();
                    }
                    update_run_time_cache = false;
                }
            }


#ifdef CACHE_MODE
            if(!record_uncached.empty()){
                UncachedList new_list;
                new_list.q = cur_q; 
                new_list.list_pos = std::move(record_uncached); 
                uncached_lists.push_back(std::move(new_list));
            }
#endif
        }
}

namespace{
    inline bool skip_list(float base_dis, float dis, float prune_factor, faiss::MetricType mt = METRIC_L2){
        if(mt == METRIC_L2){
            if(dis > prune_factor * base_dis)
                return true;
            return false;
        }else{
            if(dis < base_dis / prune_factor)
                return true;
            return false;
        }
    }
}



void IndexDiskV::search_uncached(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler,
    UncachedLists& uncached_lists
    )const {
    if (n == 0) {
        return;
    }

    size_t uncached_query_num = uncached_lists.size();
    if(uncached_query_num == 0){
        return;
    }

    std::vector<DiskInvlist> disk_invlists;
    int max_lists = 0;
    for(int i = 0; i < uncached_query_num; i++){
        if(uncached_lists[i].list_pos.size() > max_lists )
            max_lists = uncached_lists[i].list_pos.size();
    }

    disk_invlists.resize(max_lists);
    std::vector<std::vector<float>> pq_distances;
    std::vector<std::vector<idx_t>> pq_ids;

    pq_distances.resize(max_lists);
    pq_ids.resize(max_lists);

    BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);

    size_t pq_code_size = this->pq.code_size;
    for(int i = 0; i < uncached_query_num; i++){

        size_t q = uncached_lists[i].q;

        const float* current_query = x + q*d;

        size_t lists_per_query = uncached_lists[i].list_pos.size();
        idx_t* list_pos = uncached_lists[i].list_pos.data();
        float* heap_sim = distances + q*k;
        idx_t* heap_ids = labels + q*k;
        auto disk_time_start = std::chrono::high_resolution_clock::now();
        AsyncRequests_IndexInfo requests_info;
        requests_info.info_requests.reserve(lists_per_query);
        int prune_threshold = lists_per_query;
        for(int j = 0; j < lists_per_query; j++){
            size_t list_no = keys[q*nprobe + list_pos[j]];
            if(skip_list(coarse_dis[q*nprobe], coarse_dis[q*nprobe + list_pos[j]], prune_factor, this->centroid_index->metric_type)){
                prune_threshold = j;
                break;
            }
            Aligned_Invlist_Info* aii = aligned_inv_info + list_no;
            size_t page_num = aii->page_count;
            size_t list_size = aii->list_size;
            std::uint64_t m_readSize = page_num*PAGE_SIZE;
            std::uint64_t m_offset = aii->page_start*PAGE_SIZE;
            requests_info.info_requests.emplace_back(page_num, m_offset, m_readSize, nullptr, list_no, list_size);
        }

        diskIOprocessor->disk_io_info_async(requests_info);
        diskIOprocessor->submit(-2);
        for(int j = 0; j < prune_threshold; j++){
            AsyncRequest_IndexInfo& request = requests_info.info_requests[j];
            disk_invlists[j].set(request.m_buffer, request.list_size, pq_code_size);
        }
        auto disk_time_end = std::chrono::high_resolution_clock::now();
        indexDiskV_stats.disk_uncache_info_elapsed+=disk_time_end - disk_time_start;

        auto pq_time_start = std::chrono::high_resolution_clock::now();
        if(true)
        {
            pqdecoder->set_query(x + q*d);
            for(int j = 0; j < prune_threshold; j++){
                idx_t list_no = keys[q*nprobe + list_pos[j]];
                size_t list_size = this->aligned_inv_info[list_no].list_size;
                decode_pq_list(pqdecoder, nullptr, list_no, coarse_dis[q*nprobe + list_pos[j]],
                                list_size, &disk_invlists[j], pq_distances[j], pq_ids[j]);
            }
        }
        auto pq_time_end = std::chrono::high_resolution_clock::now();
        indexDiskV_stats.pq_uncache_elapsed+=pq_time_end - pq_time_start;
        AsyncReadRequests_Partial_PQDecode request_p;
        for(int j = 0; j < lists_per_query; j++){
            idx_t list_no = keys[q*nprobe + list_pos[j]];
            size_t list_size = this->aligned_inv_info[list_no].list_size;
            if(skip_list(coarse_dis[q*nprobe], coarse_dis[q*nprobe + list_pos[j]], prune_factor, this->centroid_index->metric_type)){
                break;
            }

            std::vector<int> vector_to_search;
            std::vector<int> vector_to_submit;
            auto time_start = std::chrono::high_resolution_clock::now();
            float* dis_line = pq_distances[j].data();

            if(pq_distances[j].size() != list_size){
                FAISS_THROW_MSG("UncachedList: pq_size and list_size are not equal");
            }
            if(disk_invlists[j].get_size() != list_size){
                FAISS_THROW_MSG("UncachedList: pq_size and list_size are not equal");
            }

            int reserve_size = 0;
            if (this->metric_type == METRIC_L2) {
                for (size_t kk = 0; kk < list_size; kk++) {
                    if (dis_line[kk] < heap_sim[0] * estimate_factor_partial) {
                        reserve_size++;
                    }
                }
            } else {
                for (size_t kk = 0; kk < list_size; kk++) {
                    if (dis_line[kk] > heap_sim[0] / estimate_factor_partial) {
                        reserve_size++;
                    }
                }
            }

            if(reserve_size == 0){
                continue;
            }
            faiss::indexDiskV_stats.searched_lists++;

            vector_to_search.reserve(reserve_size + 10);
            vector_to_submit.reserve(reserve_size + 10);

            if (this->metric_type == METRIC_L2) {
                for (size_t kk = 0; kk < list_size; kk++) {
                    if (dis_line[kk] < heap_sim[0] * estimate_factor_partial) {
                        vector_to_search.push_back(static_cast<int>(kk));
                        vector_to_submit.push_back(static_cast<int>(kk));
                    }
                }
            } else {
                for (size_t kk = 0; kk < list_size; kk++) {
                    if (dis_line[kk] > heap_sim[0] / estimate_factor_partial) {
                        vector_to_search.push_back(static_cast<int>(kk));
                        vector_to_submit.push_back(static_cast<int>(kk));
                    }
                }
            }

            if(vector_to_search.empty()){
                continue;
            }

            size_t vectors_num = vector_to_search.size();
            if(vectors_num <= 1){
                continue;
            }

            auto time_end = std::chrono::high_resolution_clock::now();
            indexDiskV_stats.memory_uncache_elapsed+=time_end - time_start;
            time_start = std::chrono::high_resolution_clock::now();

            std::vector<Page_to_Search> page_to_search(vectors_num);
            std::vector<size_t> vec_page_proj(vectors_num);

            int num_page_to_search = diskIOprocessor->process_page_transpage(
                    vector_to_submit.data(),
                    page_to_search.data(),
                    vec_page_proj.data(),
                    vectors_num);

            Aligned_Cluster_Info* cluster_info = &aligned_cluster_info[list_no];
            const int per_page_element = diskIOprocessor->get_per_page_element();
            const int per_page_vector = per_page_element/d;

            std::vector<PageSegment> merged_segments;
            int max_continous_pages = 2;
            merged_segments.reserve(num_page_to_search);

            if (num_page_to_search > 0) {
                merge_pages_transpage(
                        merged_segments,
                        page_to_search.data(),
                        vector_to_submit.data(),
                        vector_to_search.data(),
                        vec_page_proj.data(),
                        vectors_num,
                        num_page_to_search,
                        per_page_element,
                        per_page_vector,
                        d,
                        list_no,
                        max_continous_pages);
            }else{
                continue;
            }
            {                                                                 
                faiss::indexDiskV_stats.searched_vector_partial += vectors_num;
                for(int seg = 0; seg<merged_segments.size(); seg++){
                    faiss::indexDiskV_stats.searched_page_partial += merged_segments[seg].page_count;
                }
                faiss::indexDiskV_stats.requests_partial += merged_segments.size();
            }

            size_t global_start = cluster_info->page_start;                    
            size_t prepare_size = request_p.list_requests.size();
            request_p.list_requests.reserve(prepare_size + merged_segments.size());

            for (size_t kk = 0; kk < merged_segments.size(); ++kk) {
                const auto& segment = merged_segments[kk];

                size_t page_num = segment.page_count;
                size_t offset = (global_start + segment.start_page) * PAGE_SIZE;

                size_t total_vector_num = page_num * per_page_vector;   // beam search
                size_t begin_idx = segment.start_page * per_page_vector;
                if(begin_idx + total_vector_num > list_size){        
                    total_vector_num = list_size - begin_idx;
                }
                const size_t* map = nullptr;
                const idx_t* ids = disk_invlists[j].get_ids();

                int* in_buffer_begin = segment.in_buffer_offsets;
                int* in_buffer_end = segment.in_buffer_offsets + segment.length;

                size_t* in_ids_start = segment.in_buffer_ids;
                size_t* in_ids_end = segment.in_buffer_ids + segment.length;

                request_p.list_requests.emplace_back(page_num, total_vector_num, begin_idx, offset, 0,map, ids,
                        in_buffer_begin, in_buffer_end, in_ids_start, in_ids_end);
            }

            time_end = std::chrono::high_resolution_clock::now();
            indexDiskV_stats.memory_uncache_elapsed+=time_end - time_start;

        }
        request_p.cal_callback = [&](AsyncRequest_Partial* requested, void* buffer){
            auto time_start = std::chrono::high_resolution_clock::now();
            int* element_offsets = requested->in_buffer_offsets.data();
            size_t* element_ids = requested->in_buffer_ids.data();

            float distance;
            const idx_t * list_ids = requested->ids;

            std::vector<float> float_vector(d);

#ifndef BLOCK_BEAM_SEARCH
            for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                if(this->metric_type == METRIC_L2)
                    distance = fvec_L2sqr(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                else
                    distance = fvec_inner_product(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                heap_handler->add(q, distance, list_ids[element_ids[m]]);
            }
#else
            for(int m = 0; m < requested->vectors_num; m++){
                if(this->metric_type == METRIC_L2)
                    distance = fvec_L2sqr(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d), d);
                else
                    distance = fvec_inner_product(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d), d);
                heap_handler->add(q, distance, list_ids[requested->begin_idx + m]);
            }
#endif
            auto time_end = std::chrono::high_resolution_clock::now();
            indexDiskV_stats.rank_uncache_elapsed+=time_end - time_start;
        };

        request_p.pq_callback = [&](){

        };
        request_p.dp_callback = [&](){

        };

        diskIOprocessor->disk_io_partial_async_pq(request_p);
        auto time_start = std::chrono::high_resolution_clock::now();
        diskIOprocessor->submit(); 
        auto time_end = std::chrono::high_resolution_clock::now();
        indexDiskV_stats.disk_uncache_calc_elapsed+=time_end - time_start;

        requests_info.page_buffers.clear();
    }
}

void IndexDiskV::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {

    auto time_start = std::chrono::high_resolution_clock::now();      // time begin

    FAISS_THROW_IF_NOT(k > 0);            

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);          

    const size_t top_cluster = this->top;    
    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;
    const float p_factor = this->prune_factor;
    auto time_end = std::chrono::high_resolution_clock::now();       // time end
    DiskResultHandler* rh = get_result_handler(n, k, distances, labels, this->metric_type);

    int thread_id = omp_get_thread_num();
    DiskIOProcessor* local_processor = diskIOprocessors[thread_id].get();

#ifdef CACHE_MODE
    UncachedLists ul;
    search_o(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh, ul);
    if(!vector_cache_setting_mode){
        search_uncached(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh, ul);
    }
    
#else
    
    search_o(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh);
#endif

    auto d_time_start = std::chrono::high_resolution_clock::now();
    rh->end();
    auto d_time_end = std::chrono::high_resolution_clock::now();
    indexDiskV_stats.delete_elapsed += d_time_end - d_time_start;
}

namespace{
    template <typename ValueType>
    DiskIOProcessor* get_DiskIOBuildProcessor_2(std::string& disk_path, size_t d, size_t ntotal, faiss::MetricType metric_type = METRIC_L2){
        return new IVF_DiskIOBuildProcessor<ValueType>(disk_path, d, ntotal, metric_type);

    }
    template <typename ValueType>
    DiskIOProcessor* get_DiskIOSearchProcessor_2(const std::string& disk_path, const size_t d) {
    #ifndef USING_ASYNC
        return new IVF_DiskIOSearchProcessor<ValueType>(disk_path, d);
    #else
        return new IVF_DiskIOSearchProcessor_Async_PQ<ValueType>(disk_path, d);
    #endif
    }
}

DiskIOProcessor* IndexDiskV::get_DiskIOBuildProcessor() {
    if(this->valueType == "float"){
         return get_DiskIOBuildProcessor_2<float>(this->disk_path, d, ntotal, this->metric_type);
    }else if(this->valueType == "uint8"){
        return get_DiskIOBuildProcessor_2<uint8_t>(this->disk_path, d, ntotal, this->metric_type);
    }else if(this->valueType == "int16"){
         return get_DiskIOBuildProcessor_2<int16_t>(this->disk_path, d, ntotal, this->metric_type);
    }else if(this->valueType == "int8"){
         return get_DiskIOBuildProcessor_2<int8_t>(this->disk_path, d, ntotal, this->metric_type);
    }
    else{
        FAISS_THROW_MSG("Unsupported type");
    }
}

DiskIOProcessor* IndexDiskV::get_DiskIOSearchProcessor() const{
    if(this->valueType == "float"){
        return get_DiskIOSearchProcessor_2<float>(this->disk_path, d);
    }else if(this->valueType == "uint8"){
        return get_DiskIOSearchProcessor_2<uint8_t>(this->disk_path, d);
    }else if(this->valueType == "int16"){
        return get_DiskIOSearchProcessor_2<int16_t>(this->disk_path, d);
    }else if(this->valueType == "int8"){
        return get_DiskIOSearchProcessor_2<int8_t>(this->disk_path, d);
    }else{
        FAISS_THROW_FMT("Unsupported type %s", this->valueType.c_str());
    }
}

IndexDiskVStats indexDiskV_stats;
void IndexDiskVStats::reset() {
    memset(this, 0, sizeof(*this));
}
}
