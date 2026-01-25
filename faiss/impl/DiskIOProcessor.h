#ifndef FAISS_DISKIOPROCESSOR_H
#define FAISS_DISKIOPROCESSOR_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <functional>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstddef>
#include <stack>
#include <type_traits>

#include <faiss/impl/FaissAssert.h>
#include <faiss/MetricType.h>
#include <faiss/invlists/InvertedLists.h>
#include <cmath> 
#include <faiss/impl/AsyncIOExtension.h>
#include <faiss/impl/DiskIOStructure.h>
#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace faiss{

struct Rerank_Info{
    std::shared_ptr<std::vector<float>> list_sim;
    std::shared_ptr<std::vector<idx_t>> list_ids;
    float* heap_sim;
    idx_t* heap_ids;

    size_t k;
    idx_t key;

    float factor_fully;
    float factor_partially;

    float* query;

    const idx_t* ids;

};

struct AsyncReadRequest
{
    std::uint64_t m_offset;
    std::uint64_t m_readSize;
    char* m_buffer;
    float* converted_buffer;
    std::function<void(AsyncReadRequest*)> m_callback;
    int m_status;    // multi-file control, 16 bits for status
    size_t len;      // help create a buffer to store float data
    
    // use it when search partially
    // size_t* in_page_offset;
    size_t D;
    Rerank_Info rerank_info;
    // Carry items like counter for callback to process.
    void* m_payload;
    bool m_success;
    // Carry exension metadata needed by some DiskIO implementations
    void* m_extension;

    AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_status(0), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
};


struct AsyncReadRequest_Partial_PQDecode{
    std::uint64_t m_offset;
    std::uint64_t m_readSize;

    // offset from at the begining of the PAGE. Counted by element, not vectors. 
    // Every offset represent a vector.
    std::vector<int> in_buffer_offsets;
    std::vector<size_t> in_buffer_ids; 

    // must be aligned
    char* m_buffer;

    float* converted_buffer;


    size_t len;      // help create a buffer to store float data
    size_t D;
    idx_t list_no;


    // necessary?
    Rerank_Info rerank_info;

    // Carry items like counter for callback to process.
    void* m_payload;
    bool m_success;
    // Carry exension metadata needed by some DiskIO implementations
    void* m_extension;

    std::function<void(AsyncReadRequest_Partial_PQDecode* requested)> callback_calculation;
    std::function<void()> callback_pqdecode;
};


struct DiskIOProcessor{
    std::string disk_path;
    size_t d;
    
    size_t total_page;

    faiss::MetricType metric_type = METRIC_L2;

    bool verbose = true;

    DiskIOProcessor(std::string disk_path, size_t d, size_t total_page = 0): disk_path(disk_path), d(d), total_page(total_page){
    }

    virtual ~DiskIOProcessor(){
        
    }

    virtual bool align_cluster_page(size_t* clusters,
                                size_t* len,
                                Aligned_Cluster_Info* acInfo,
                                size_t nlist){
                                    FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
                                    return true;
                                };

    virtual bool reorganize_vectors(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            std::vector<std::vector<faiss::idx_t>> ids){
                            FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
                            return true;

                            };
    virtual bool reorganize_vectors_2(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            bool do_in_list_cluster){
                                return true;
                            }

    // if we store the all the vectors in memory when building the index(Not recommend)
    virtual bool reorganize_vectors_in_memory(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            ArrayInvertedLists* build_invlists,
                            bool do_in_list_cluster,
                            bool keep_disk = false){
                                return true;
                            }


    virtual size_t reorganize_list(
        Index& quantizer, 
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Cluster_Info* acInfo,
        size_t* clusters,
        size_t* len,
        size_t nlist
    ){
        FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
        return 0;         
    }

    virtual bool organize_select_list(
        size_t pq_size,
        size_t entry_size,
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Invlist_Info* invInfo, 
        size_t nlist,
        std::string select_lists_path
    ){
        FAISS_THROW_MSG("Func organize_select_list: Processor Base Class does not support write operation");
        return false;  
    }
    
    virtual void disk_io_all(int D,
                            size_t len,
                            size_t listno, 
                            float* vectors,
                            Aligned_Cluster_Info* acInfo){
        FAISS_THROW_MSG("Do not call virtual function disk_io_all!");
    }

    virtual void disk_io_single(int D,
                                size_t len,
                                size_t listno,
                                size_t nth,
                                float* vector,
                                Aligned_Cluster_Info* acInfo){
        
    }

    virtual void disk_io_all_async(std::shared_ptr<AsyncReadRequest>& asyncReadRequest){
        FAISS_THROW_MSG("Base IO processor should not be used : disk_io_all_async");
    }

    virtual void submit(int num = -1){
        FAISS_THROW_MSG("Base IO processor should not be used : submit");
    }

    virtual void initial(std::uint64_t maxIOSize = (1 << 20),
                        std::uint32_t maxReadRetries = 2,
                        std::uint32_t maxWriteRetries = 2,
                        std::uint16_t threadPoolSize = 4){
        FAISS_THROW_MSG("Do not call virtual function initial!");
    }

    virtual int process_page(int* vector_to_submit, int* page_to_search, size_t* vec_page_proj, size_t len_p){
            FAISS_THROW_MSG("Func process_page: Processor Base Class does not support this operation");
                            
            return -1;
    }

    virtual int process_page_transpage(int* vector_to_submit, Page_to_Search* page_to_search, size_t* vec_page_proj, size_t len_p){
            FAISS_THROW_MSG("Func process_page_transpage: Processor Base Class does not support this operation");
                            
            return -1;
    }

    virtual void disk_io_partial_async_pq(AsyncReadRequests_Partial_PQDecode& asyncReadRequests_p){
        
    }

    virtual void disk_io_full_async_pq(AsyncReadRequests_Full_PQDecode& asyncReadRequests_f){
        
    }

    virtual void disk_io_info_async(AsyncRequests_IndexInfo& asyncReadRequests_i){

    }
    
    virtual void convert_to_float(size_t n, float* vectors, void* disk_data){

    }

    virtual float* convert_to_float_single(float* vectors, void* disk_data, int begin){
        FAISS_THROW_MSG("Func convert_to_float_single: Processor Base Class does not support this operation");
        return nullptr;
    }

    virtual void test(){
        std::cout << "DiskIOBase:" << std::endl;
    }

    virtual int get_per_page_element(){
        return 0;
    }

    virtual int shut_down(){
        FAISS_THROW_MSG("Do not call virtual function shutdown!");
        return 0;
    }
    
};


namespace{

    void in_list_pq_ids_reassign(ClusteredArrayInvertedLists* c_array_invlists, size_t list_no){
        const size_t* map = c_array_invlists->get_inlist_map(list_no);
        if (!map) return;
        auto& codes = c_array_invlists->codes[list_no];
        auto& ids = c_array_invlists->ids[list_no];
        size_t list_size = ids.size();
        size_t code_size = c_array_invlists->code_size;

        if (list_size == 0) return; 

        std::vector<uint8_t> new_codes(list_size * code_size);
        std::vector<idx_t> new_ids(list_size);

        const uint8_t* codes_ptr = codes.data();
        const idx_t* ids_ptr = ids.data();

        for (size_t i = 0; i < list_size; ++i) {
            size_t new_pos = map[i];
            if (new_pos >= list_size) {
                throw std::runtime_error("Invalid map index");
            }
            std::copy(
                codes_ptr + i * code_size,
                codes_ptr + (i + 1) * code_size,
                new_codes.begin() + new_pos * code_size
            );
            new_ids[new_pos] = ids_ptr[i];
        }
        codes = MaybeOwnedVector<uint8_t>(std::move(new_codes));
        ids = MaybeOwnedVector<idx_t>(std::move(new_ids));
    }

    void in_list_map_reassign(
        size_t n, 
        float* nx, 
        Index& index, 
        ClusteredArrayInvertedLists* invlists, 
        size_t list_no){
        
        assert(n!=0);
        assert(index.ntotal != 0);
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * 1]);
        index.assign(n, nx, coarse_idx.get(), 1);
        std::vector<float> reordered_nx(n * index.d);
        std::vector<size_t> cluster_counts(index.ntotal, 0);
        for (size_t i = 0; i < n; ++i) {
            cluster_counts[coarse_idx[i]]++;
        }
        std::vector<size_t> cluster_offsets(index.ntotal, 0);
        size_t cumulative_offset = 0;
        for (size_t cluster_id = 0; cluster_id < index.ntotal; ++cluster_id) {
            cluster_offsets[cluster_id] = cumulative_offset;
            cumulative_offset += cluster_counts[cluster_id];
        }

        for (size_t i = 0; i < n; ++i) {
            size_t cluster_id = coarse_idx[i];
            size_t new_pos = cluster_offsets[cluster_id]++;
            invlists->updata_inlist_map(list_no, i, new_pos);
            std::memcpy(&reordered_nx[new_pos * index.d], &nx[i * index.d], index.d * sizeof(float));
        }
        std::memcpy(nx, reordered_nx.data(), n * index.d * sizeof(float));

    }

    std::vector<idx_t> sort_vectors_by_proximity(size_t n, idx_t* labels, size_t candidate_num) {
        std::vector<bool> visited(n, false);
        std::vector<idx_t> new_order;
        new_order.reserve(n);

        std::stack<std::pair<idx_t, size_t>> stack;
        idx_t start_node = 0;
        visited[start_node] = true;
        new_order.push_back(start_node);
        stack.push({start_node, 1});

        while (!stack.empty()) {
            auto current_pair = stack.top();
            stack.pop();

            idx_t current = current_pair.first;
            size_t j = current_pair.second;
            bool found = false;
            while (j < candidate_num) {
                idx_t neighbor = labels[current * candidate_num + j];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    new_order.push_back(neighbor);
                    if(j + 1 < candidate_num)
                        stack.push({current, j + 1});
                    stack.push({neighbor, 1});
                    found = true;
                    break;
                }
                j++;
            }

            if (!found && stack.empty() && new_order.size() < n) {
                for (idx_t i = 0; i < n; ++i) {
                    if (!visited[i]) {
                        visited[i] = true;
                        new_order.push_back(i);
                        stack.push({i, 1});
                        break;
                    }
                }
            }
        }

        return new_order;
    }

    std::vector<idx_t> in_list_centroid_reassign(Index& index, size_t num_clusters){
        size_t total = index.ntotal;
        size_t d = index.d;

        size_t candidates_num = std::min((size_t)4, total);

        std::vector<float> dis(total*candidates_num);
        std::vector<idx_t> ids(total*candidates_num);

        std::vector<float> base_vector(d*total);
        index.reconstruct_n(0, total, base_vector.data());
        index.search(total, base_vector.data(), candidates_num, dis.data(), ids.data());

        auto new_order = sort_vectors_by_proximity(total, ids.data(), candidates_num);
        std::vector<float> reordered_centroids(total * d);
        for (size_t i = 0; i < total; ++i) {
            size_t original_index = new_order[i];
            index.reconstruct(original_index, &reordered_centroids[i * d]);
        }
        index.reset();
        index.add(total, reordered_centroids.data());
        return new_order;
    }

    std::vector<size_t> generate_inverted_map(const std::vector<idx_t>& map, size_t nlist) {
        std::vector<size_t> inverted_map(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            inverted_map[map[i]] = i;
        }
        return inverted_map;
    }
}

// The members of this class is redundent
// So, there is no need to write it in disk
// Just make it when needed( read_index and make one)

const int IN_LIST_CLUSTER_RATIO = 40;
template<typename ValueType>
struct IVF_DiskIOBuildProcessor : DiskIOProcessor{

    size_t ntotal;
    IVF_DiskIOBuildProcessor(std::string disk_path, 
                             size_t d,
                             size_t ntotal, faiss::MetricType metric_type = METRIC_L2): DiskIOProcessor(disk_path, d), ntotal(ntotal){    
                                this->metric_type = metric_type;
    }

    std::vector<ValueType> convert_to_ValueType(const std::vector<float>& data) {
        std::vector<ValueType> converted(data.size());
        std::transform(data.begin(), data.end(), converted.begin(), [](float value) {
            return static_cast<ValueType>(value);
        });
        return converted;
    }

    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        ValueType* original_vectors = reinterpret_cast<ValueType*>(disk_data);
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < d; j++)
            {
                vectors[i*d+j] = static_cast<float>(original_vectors[i*d+j]);
            }
        }
    }

    bool align_cluster_page(size_t* clusters,
                        size_t* len,
                        Aligned_Cluster_Info* acInfo,
                        size_t nlist) override {
        
        size_t cumulative_pages = 0; 

        for (size_t i = 0; i < nlist; ++i) {
            size_t total_bytes = len[i] * d * sizeof(ValueType);
            size_t page_count = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

            size_t aligned_bytes = page_count * PAGE_SIZE;

            size_t padding_offset = aligned_bytes - total_bytes;

            acInfo[i].page_start = cumulative_pages;
            acInfo[i].page_count = page_count;
            acInfo[i].padding_offset = padding_offset;

            cumulative_pages += page_count;
        }

        this->total_page = cumulative_pages;

        std::cout << "Page alignment completed!" << std::endl;
        return true;
    }

    bool reorganize_vectors(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            std::vector<std::vector<faiss::idx_t>> ids) override {
        idx_t old_total = this->ntotal - n;
        std::cout << "old_total = "  << old_total << "  ntotal = " << this->ntotal << " n = " << n << std::endl;

        if (old_clusters == nullptr && old_len == nullptr) {
            align_cluster_page(clusters, len, acInfo, nlist);
            std::string clustered_disk_path = disk_path + ".clustered";

            std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open clustered disk file for writing.");
            }
            for (size_t i = 0; i < nlist; ++i) {
                size_t count = len[i];
                for (size_t j = 0; j < count; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[id * d];
                    std::vector<ValueType> converted_vector(d);
                    for (size_t k = 0; k < d; ++k) {
                        converted_vector[k] = static_cast<ValueType>(vector[k]); 
                    }
                    disk_data_write.write(reinterpret_cast<const char*>(converted_vector.data()), d * sizeof(ValueType));
                }
                size_t padding = acInfo[i].padding_offset / sizeof(ValueType);
                if (padding > 0) {
                    std::vector<ValueType> padding_data(padding, 0);
                    disk_data_write.write(reinterpret_cast<const char*>(padding_data.data()), padding * sizeof(ValueType));
                }
            }

            disk_data_write.close();
            std::cout << "Reorganize vectors: Initial write completed with page alignment." << std::endl;
            return true;
        } else {
            std::vector<Aligned_Cluster_Info> old_acInfo(nlist);
            for (size_t i = 0; i < nlist; ++i) {
                old_acInfo[i] = acInfo[i];
            }
            align_cluster_page(clusters, len, acInfo, nlist);

            std::string tmp_disk = disk_path + ".tmp";
            int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
            if(file_result == 0)
                std::cout << "Successfully renamed to: " << tmp_disk << std::endl;
            else {
                std::cerr << "Failed to rename to: " << tmp_disk << std::endl;
                throw std::runtime_error("Failed to rename disk file.");
            }

            std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
            if (!temp_disk_read.is_open()) {
                throw std::runtime_error("Failed to open temporary disk file for reading.");
            }

            std::ofstream disk_data_write(disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open clustered disk file for writing.");
            }

            for (size_t i = 0; i < nlist; ++i) {
                size_t old_count = old_len[i];
                size_t new_count = len[i];
                size_t total_count = len[i];

                std::cout << " old_len = " << old_len[i] << "  len = " << len[i] << std::endl;

                size_t total_bytes = total_count * d * sizeof(ValueType);
                std::vector<ValueType> combined_data(total_count * d, 0);
                if (old_count > 0) {
                    size_t old_bytes = old_count * d * sizeof(ValueType);
                    size_t old_offset_bytes = old_acInfo[i].page_start * PAGE_SIZE;
                    temp_disk_read.seekg(old_offset_bytes, std::ios::beg);
                    temp_disk_read.read(reinterpret_cast<char*>(combined_data.data()), old_bytes);
                    if (temp_disk_read.gcount() != static_cast<std::streamsize>(old_bytes)) {
                        throw std::runtime_error("Failed to read complete old cluster data.");
                    }
                }

                for (size_t j = old_count; j < new_count ; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[(id-old_total) * d];  // new 
                    for (size_t k = 0; k < d; ++k) {
                        combined_data[j * d + k] = static_cast<ValueType>(vector[k]);
                    }
                }

                disk_data_write.write(reinterpret_cast<const char*>(combined_data.data()), total_count * d * sizeof(ValueType));

                if(verbose)
                {   
                    size_t written_pages = (total_count * d * sizeof(ValueType) + PAGE_SIZE - 1) / PAGE_SIZE;
                    size_t expected_pages = acInfo[i].page_count;
                }

                size_t new_padding = acInfo[i].padding_offset;
                if (new_padding > 0) {
                    std::vector<ValueType> padding_data(new_padding / sizeof(ValueType), 0);
                    disk_data_write.write(reinterpret_cast<const char*>(padding_data.data()), new_padding);
                }
            }
            disk_data_write.close();
            temp_disk_read.close();

            std::remove(tmp_disk.c_str());

            std::cout << "Reorganize vectors: Merged old and new data with page alignment." << std::endl;
            return false;
        }
    }

    bool reorganize_vectors_2(idx_t n, 
                          const float* data, 
                          size_t* old_clusters, 
                          size_t* old_len,
                          size_t* clusters,
                          size_t* len,
                          Aligned_Cluster_Info* acInfo,
                          size_t nlist,
                          ClusteredArrayInvertedLists* c_array_invlists,
                          bool do_in_list_cluster) override 
{
    idx_t old_total = this->ntotal - n;
    if (old_clusters == nullptr && old_len == nullptr) {

        align_cluster_page(clusters, len, acInfo, nlist);
        std::string clustered_disk_path = disk_path + ".clustered";

        std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
        if (!disk_data_write.is_open()) {
            throw std::runtime_error("Failed to open clustered disk file for writing.");
        }
        std::vector< std::vector<ValueType> > all_list_buffers(nlist);  
        std::vector<size_t> list_buffer_sizes(nlist, 0);

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nlist; ++i) {
            size_t count = len[i];
            std::vector<float> vectors(count * d);
            for (size_t j = 0; j < count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                std::copy(&data[id * d],
                          &data[(id + 1) * d],
                          vectors.data() + j * d);
            }
            if (do_in_list_cluster) {
                size_t in_list_cluster_num = count / IN_LIST_CLUSTER_RATIO;
                if (in_list_cluster_num > 1) {
                    Clustering clus(d, in_list_cluster_num);
                    IndexFlat assigner(d, this->metric_type);
                    clus.train(count, vectors.data(), assigner);
                    in_list_centroid_reassign(assigner, in_list_cluster_num);
                    in_list_map_reassign(count, vectors.data(), assigner, c_array_invlists, i);
                    in_list_pq_ids_reassign(c_array_invlists, i);
                } else {
                    // If too few vectors, just keep them in the same order.
                    for (size_t m = 0; m < count; m++) {
                        c_array_invlists->updata_inlist_map(i, m, m);
                    }
                }
            }
            auto converted_vectors = convert_to_ValueType(vectors);
            size_t padding_vals = acInfo[i].padding_offset / sizeof(ValueType);
            size_t total_vals   = converted_vectors.size() + padding_vals;

            all_list_buffers[i].resize(total_vals, 0);
            std::memcpy(all_list_buffers[i].data(),
                        converted_vectors.data(),
                        converted_vectors.size() * sizeof(ValueType));
            list_buffer_sizes[i] = total_vals; 
        } 

        for (size_t i = 0; i < nlist; ++i) {
            if (!all_list_buffers[i].empty()) {
                disk_data_write.write(
                    reinterpret_cast<const char*>(all_list_buffers[i].data()),
                    list_buffer_sizes[i] * sizeof(ValueType));
            }
        }

        disk_data_write.close();
        std::cout << "Reorganize vectors: Initial write completed with page alignment." << std::endl;
        return true;
    }
    else {
        std::vector<Aligned_Cluster_Info> old_acInfo(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            old_acInfo[i] = acInfo[i];
        }
        align_cluster_page(clusters, len, acInfo, nlist);

        std::string tmp_disk = disk_path + ".tmp";
        int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
        if (file_result == 0) {
            std::cout << "Successfully renamed to: " << tmp_disk << std::endl;
        } else {
            std::cerr << "Failed to rename to: " << tmp_disk << std::endl;
            throw std::runtime_error("Failed to rename disk file.");
        }

        std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
        if (!temp_disk_read.is_open()) {
            throw std::runtime_error("Failed to open temporary disk file for reading.");
        }
        std::ofstream disk_data_write(disk_path, std::ios::binary);
        if (!disk_data_write.is_open()) {
            throw std::runtime_error("Failed to open clustered disk file for writing.");
        }

        std::vector< std::vector<ValueType> > all_list_buffers(nlist);
        std::vector<size_t> list_buffer_sizes(nlist, 0);

        std::vector<size_t> old_offsets(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            old_offsets[i] = old_acInfo[i].page_start * PAGE_SIZE;
        }

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nlist; ++i) {
            size_t old_offset = old_offsets[i];
            size_t old_count  = old_len[i];
            size_t new_count  = len[i] - old_count;
            size_t count      = len[i];

            std::vector<ValueType> disk_vectors(old_count * d, 0);
            {

                std::ifstream local_in(tmp_disk, std::ios::binary);
                if (!local_in.is_open()) {
                    throw std::runtime_error("Thread failed to open tmp file for reading.");
                }
                local_in.seekg(old_offset, std::ios::beg);
                local_in.read(reinterpret_cast<char*>(disk_vectors.data()),
                              old_count * d * sizeof(ValueType));
                local_in.close();
            }

            // Convert the old vectors to float for potential re-clustering
            std::vector<float> all_vectors(count * d, 0.0f);
            convert_to_float(old_count, all_vectors.data(), disk_vectors.data());

            for (size_t j = old_count; j < count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                std::copy(&data[(id - old_total) * d],
                          &data[(id - old_total + 1) * d],
                          all_vectors.data() + j * d);
            }
            if (do_in_list_cluster) {
                size_t in_list_cluster_num = count / IN_LIST_CLUSTER_RATIO;
                if (in_list_cluster_num > 1) {
                    Clustering clus(d, in_list_cluster_num);
                    IndexFlat assigner(d, this->metric_type);
                    clus.train(count, all_vectors.data(), assigner);
                    in_list_centroid_reassign(assigner, in_list_cluster_num);
                    in_list_map_reassign(count, all_vectors.data(), assigner, c_array_invlists, i);
                    in_list_pq_ids_reassign(c_array_invlists, i);
                } else {
                    for (size_t m = 0; m < count; m++) {
                        c_array_invlists->updata_inlist_map(i, m, m);
                    }
                }
            }

            auto converted_vectors = convert_to_ValueType(all_vectors);
            size_t new_padding = acInfo[i].padding_offset;
            size_t padding_vals = new_padding / sizeof(ValueType);

            size_t total_vals = converted_vectors.size() + padding_vals;
            all_list_buffers[i].resize(total_vals, 0);
            std::memcpy(all_list_buffers[i].data(),
                        converted_vectors.data(),
                        converted_vectors.size() * sizeof(ValueType));

            list_buffer_sizes[i] = total_vals;
        } 

        // Single-threaded final writes
        for (size_t i = 0; i < nlist; ++i) {
            if (!all_list_buffers[i].empty()) {
                disk_data_write.write(
                    reinterpret_cast<const char*>(all_list_buffers[i].data()),
                    list_buffer_sizes[i] * sizeof(ValueType));
            }
        }

        disk_data_write.close();
        temp_disk_read.close();
        std::remove(tmp_disk.c_str());

        std::cout << "Reorganize vectors: Merged old and new data with page alignment." << std::endl;
        return false;
    }
}

    
    bool reorganize_vectors_in_memory(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            ArrayInvertedLists* build_invlists,
                            bool do_in_list_cluster,
                            bool keep_disk = false)override
    {
        idx_t old_total = this->ntotal - n;
        if(1)
        {    
            if(keep_disk){
                align_cluster_page(clusters, len, acInfo, nlist);
                std::string clustered_disk_path = disk_path + ".clustered";

                std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
                if (!disk_data_write.is_open()) {
                    throw std::runtime_error("Failed to open clustered disk file for writing.");
                }

                std::vector< std::vector<ValueType> > all_list_buffers(nlist);  
                std::vector<size_t> list_buffer_sizes(nlist, 0);

#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < nlist; ++i) {
                    std::cout << "clustering " << i << "\n";
                    size_t count = len[i];
                    std::vector<float> vectors(count * d);

                    const float* original_data = (const float*)build_invlists->get_codes(i);

                    std::copy(original_data, original_data + count*d, vectors.data());

                    if (do_in_list_cluster) {
                        size_t in_list_cluster_num = count / 32;
                        if (in_list_cluster_num > 1) {
                            Clustering clus(d, in_list_cluster_num);
                            IndexFlatL2 assigner(d);
                            clus.train(count, vectors.data(), assigner);
                            in_list_centroid_reassign(assigner, in_list_cluster_num);
                            in_list_map_reassign(count, vectors.data(), assigner, c_array_invlists, i);
                            in_list_pq_ids_reassign(c_array_invlists, i);
                        } else {
                            // If too few vectors, just keep them in the same order.
                            for (size_t m = 0; m < count; m++) {
                                c_array_invlists->updata_inlist_map(i, m, m);
                            }
                        }
                    }

                    // Convert floats -> ValueType
                    auto converted_vectors = convert_to_ValueType(vectors);
                    std::cout << "Converted to original format...\n";

                    // Prepare final buffer for this list:
                    //   (vectors + any padding)
                    size_t padding_vals = acInfo[i].padding_offset / sizeof(ValueType);
                    size_t total_vals   = converted_vectors.size() + padding_vals;

                    all_list_buffers[i].resize(total_vals, 0);
                    // copy the real vector data
                    std::memcpy(all_list_buffers[i].data(),
                                converted_vectors.data(),
                                converted_vectors.size() * sizeof(ValueType));
                    list_buffer_sizes[i] = total_vals; 
                } // end parallel for

                // Step B: Single-threaded writing in correct order
                for (size_t i = 0; i < nlist; ++i) {
                    // Write out the entire buffer for list i
                    if (!all_list_buffers[i].empty()) {
                        disk_data_write.write(
                            reinterpret_cast<const char*>(all_list_buffers[i].data()),
                            list_buffer_sizes[i] * sizeof(ValueType));
                    }
                }

                disk_data_write.close();
                std::cout << "Stored in disk!" << std::endl;
                return true;
            }
            std::cout << "Skip mid clusters" << std::endl;
            return false;
        }
        else{
    
            return false;


        }
        
    }
    
    void reorganize_list_disk(
        Aligned_Cluster_Info* acInfo,
        std::vector<idx_t>& map,
        size_t nlist){
        
        std::cout << "reorganize_list_disk." << std::endl;
        std::string temp_file = disk_path + ".temp";
        if (std::rename(disk_path.c_str(), temp_file.c_str()) != 0) {
            throw std::runtime_error("Failed to rename the source file to a temporary file.");
        }
        std::ifstream source(temp_file, std::ios::binary);
        if (!source.is_open()) {
            throw std::runtime_error("Failed to open the temporary file for reading.");
        }
        std::ofstream destination(disk_path, std::ios::binary);
        if (!destination.is_open()) {
            throw std::runtime_error("Failed to open the new file for writing.");
        }
        for (size_t i = 0; i < nlist; i++) {
            size_t o = map[i];
            FAISS_THROW_IF_NOT(o < nlist);
            size_t offset = acInfo[o].page_start * PAGE_SIZE;
            size_t data_size = acInfo[o].page_count * PAGE_SIZE;
            
            source.seekg(offset, std::ios::beg);
            std::vector<char> buffer(data_size);
            source.read(buffer.data(), data_size);
            if (source.gcount() != data_size) {
                throw std::runtime_error("Failed to read data for cluster from the temporary file.");
            }

            destination.write(buffer.data(), data_size);
            if (!destination) {
                throw std::runtime_error("Failed to write data to the new file.");
            }
        }

        source.close();
        destination.close();

        if (std::remove(temp_file.c_str()) != 0) {
            throw std::runtime_error("Failed to delete the temporary file.");
        }
    }

    size_t reorganize_list(
        Index& quantizer, 
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Cluster_Info* acInfo,
        size_t* clusters,
        size_t* len,
        size_t nlist) override{
        std::cout << "permuting invlists" << std::endl;
        std::vector<idx_t> list_reorg_map = in_list_centroid_reassign(reinterpret_cast<IndexFlat&>(quantizer), nlist);

        reorganize_list_disk(acInfo, list_reorg_map, nlist);

        c_array_invlists->permute_invlists(list_reorg_map.data());

        std::cout << "updating info" << std::endl;
        size_t current_offset = 0;
        for (size_t i = 0; i < nlist; ++i) {
            clusters[i] = current_offset;
            len[i] = c_array_invlists->ids[i].size();
            current_offset += len[i];
        }
        align_cluster_page(clusters, len, acInfo, nlist);

        return true;         
    }


    bool align_list_page(size_t entry_size,   // clustered: sizeof(idx) + sizeof(pq_size)
                        size_t* len,
                        Aligned_Invlist_Info* invInfo,
                        size_t nlist) {
         
        size_t cumulative_pages = this->total_page;   

        for (size_t i = 0; i < nlist; ++i) {
            size_t total_bytes = len[i] * entry_size;
            size_t page_count = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

            size_t aligned_bytes = page_count * PAGE_SIZE;

            size_t padding_offset = aligned_bytes - total_bytes;

            invInfo[i].page_start = cumulative_pages;
            invInfo[i].page_count = page_count;
            invInfo[i].padding_offset = padding_offset;
            invInfo[i].list_size = len[i];

            cumulative_pages += page_count;
        }

        std::cout << "Inverted list pages alignment completed!" << std::endl;
        return true;
    }


    bool organize_select_list(
        size_t pq_size,
        size_t entry_size,
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Invlist_Info* invInfo, 
        size_t nlist,
        std::string select_lists_path)override
    {
        std::vector<size_t> list_size(nlist);

        for(int i = 0; i < nlist; i++){
            list_size[i] = c_array_invlists->list_size(i);
            invInfo[i].list_size = list_size[i];
        }

        align_list_page(entry_size, list_size.data(), invInfo, nlist);

        std::ofstream out_file(select_lists_path, std::ios::binary | std::ios::app);  
        //std::ofstream out_file(select_lists_path, std::ios::binary | std::ios::trunc);


        if (!out_file.is_open()) {
            std::cerr << "Failed to open file: " << select_lists_path << std::endl;
            return false;
        }

        std::cout << "pq_size:" << pq_size << "  idx_t:" << sizeof(idx_t) << "  size_t:" << sizeof(size_t) <<"\n";

        for (size_t i = 0; i < nlist; ++i) {
            size_t ids_size = list_size[i] * sizeof(idx_t);
            size_t codes_size = list_size[i] * pq_size;

            const idx_t* ids = c_array_invlists->get_ids(i);
            const uint8_t* codes = c_array_invlists->get_codes(i);

            out_file.write(reinterpret_cast<const char*>(ids), ids_size);
            out_file.write(reinterpret_cast<const char*>(codes), codes_size);

            size_t padding_bytes = invInfo[i].padding_offset;
            if (padding_bytes > 0) {
                std::vector<char> padding(padding_bytes, 0);
                out_file.write(padding.data(), padding_bytes);
            }
        }

        out_file.close();
        std::cout << "Data written successfully to " << select_lists_path << std::endl;
        return true;
    }
};

// sync form
template<typename ValueType>
struct IVF_DiskIOSearchProcessor : DiskIOProcessor{

    FILE* file_ptr;

    IVF_DiskIOSearchProcessor(std::string disk_path, 
                             size_t d): DiskIOProcessor(disk_path, d), file_ptr(nullptr){
        file_ptr = fopen(disk_path.c_str(), "rb");
        if (!file_ptr) {
            throw std::runtime_error("Failed to open disk file for reading.");
        }
    }

    ~IVF_DiskIOSearchProcessor() {
        if (file_ptr) {
            fclose(file_ptr);
            std::cout << "File closed successfully." << std::endl;
        }
    }
    
    void disk_io_all(int D,
                    size_t len,
                    size_t listno,
                    float* vectors,
                    Aligned_Cluster_Info* acInfo) override {
        
        if (!file_ptr) {
            std::cerr << "File is not open for reading!" << std::endl;
            return;
        }

        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;
        size_t page_count = acInfo[listno].page_count;

        size_t offset = page_start * PAGE_SIZE;

        size_t total_bytes = page_count * PAGE_SIZE - padding_offset;

        if (fseek(file_ptr, offset, SEEK_SET) != 0) {
            std::cerr << "Failed to seek to the required position in the file." << std::endl;
            return;
        }
        std::vector<ValueType> buffer(len * D);

        size_t read_count = fread(buffer.data(), sizeof(ValueType), total_bytes / sizeof(ValueType), file_ptr);
        if (read_count != total_bytes / sizeof(ValueType)) {
            std::cerr << "Failed to read the expected number of bytes from disk!" << std::endl;
            return;
        }
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < D; ++j) {
                vectors[i * D + j] = static_cast<float>(buffer[i * D + j]);
            }
        }


    }

    void disk_io_single(int D,
                        size_t len,
                        size_t listno,
                        size_t nth,
                        float* vector,
                        Aligned_Cluster_Info* acInfo) override {
        if (!file_ptr) {
            std::cerr << "File is not open for reading!" << std::endl;
            return;
        }
        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;

        size_t offset = page_start * PAGE_SIZE + nth * D * sizeof(ValueType);

        if (fseek(file_ptr, offset, SEEK_SET) != 0) {
            std::cerr << "Failed to seek to the required position in the file." << std::endl;
            return;
        }

        std::vector<ValueType> buffer(D);
        size_t read_count = fread(buffer.data(), sizeof(ValueType), D, file_ptr);
        if (read_count != D) {
            std::cerr << "Failed to read the expected number of bytes for the vector!" << std::endl;
            return;
        }

        for (size_t i = 0; i < D; ++i) {
            vector[i] = static_cast<float>(buffer[i]);
        }
                
    }

};

// async form
template<typename ValueType>
struct IVF_DiskIOSearchProcessor_Async_PQ : DiskIOProcessor{
    size_t factor_partial;
    
    AsyncReadRequests_Partial_PQDecode*  partial_diskRequests;
    AsyncReadRequests_Full_PQDecode* full_diskRequests;
    AsyncRequests_IndexInfo* info_diskRequests;

    int fd;
    aio_context_t aio_ctx; 

    IVF_DiskIOSearchProcessor_Async_PQ(std::string disk_path, 
                               size_t d): DiskIOProcessor(disk_path, d), aio_ctx(0){
    }

    void initial(std::uint64_t maxIOSize = (1 << 20),
                 std::uint32_t maxReadRetries = 2,
                 std::uint32_t maxWriteRetries = 2,
                 std::uint16_t threadPoolSize = 4) override {
        maxIOSize = 1<<10;
        // O_DIRECT
        fd = open(disk_path.c_str(), O_RDONLY | O_DIRECT);  
        if (fd < 0) {
            perror("Failed to open file with O_DIRECT");
            throw std::runtime_error("Failed to open file");
        }
        
        auto ret = syscall(__NR_io_setup, (int)maxIOSize, &aio_ctx);
        if(ret < 0){
            perror("io_setup failed");
            throw std::runtime_error("Failed to initialize AIO context");
        }
    }

    ~IVF_DiskIOSearchProcessor_Async_PQ() {
        if (syscall(__NR_io_destroy, aio_ctx) < 0) {
            std::cerr << "Failed to destroy AIO context." << std::endl;
        }
        if (fd >= 0) {
            close(fd);
        }
    }

#if defined(__AVX2__)
   void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        uint8_t* data = static_cast<uint8_t*>(disk_data);

        size_t i = 0;
        const size_t simd_step = 32; // Process 32 uint8_t values per iteration

        for (size_t i = 0; i < n; i += 32) {
            __m256i raw_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[i]));
            __m128i data_lo = _mm256_castsi256_si128(raw_data);
            __m128i data_hi = _mm256_extracti128_si256(raw_data, 1); 
            __m256i uint16_lo = _mm256_cvtepu8_epi16(data_lo);
            __m256i uint16_hi = _mm256_cvtepu8_epi16(data_hi);

            __m256 float_lo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(uint16_lo, _mm256_setzero_si256()));
            __m256 float_hi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(uint16_lo, _mm256_setzero_si256()));
            __m256 float_lo2 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(uint16_hi, _mm256_setzero_si256()));
            __m256 float_hi2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(uint16_hi, _mm256_setzero_si256()));

            _mm256_storeu_ps(&vectors[i + 0], float_lo);
            _mm256_storeu_ps(&vectors[i + 8], float_hi);
            _mm256_storeu_ps(&vectors[i + 16], float_lo2);
            _mm256_storeu_ps(&vectors[i + 24], float_hi2);
        }

        for (; i < n; ++i) {
            vectors[i] = static_cast<float>(data[i]);
        }
    }
#else
    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
            ValueType* original_vectors = reinterpret_cast<ValueType*>(disk_data);
            for(size_t i = 0; i < n; i++){
                for(size_t j = 0; j < d; j++)
                {
                    vectors[i*d+j] = static_cast<float>(original_vectors[i*d+j]);
                }
            }
        }
#endif
    

    float* convert_to_float_single(float* vector, void* disk_data, int begin){
        
        ValueType* original_vector = reinterpret_cast<ValueType*>(disk_data) + begin;
        for(int i = 0; i < d; i++){
            vector[i] = (float)(original_vector[i]);
        }
        return vector;
    }


    struct timespec AIOTimeout {0, 30000};
    void submit_fully(int num){
        int iocb_num = num;
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;
        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_Full_Batch* readRequest = full_diskRequests->list_requests.data() + i;
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(full_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

            iocbs.emplace_back(myiocb);

            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = false;
        int kk = 0;

        if(totalToSubmit == 0){
            if(pq_decode){
                full_diskRequests->pq_callback();
                pq_decode = false;
            }
            return;
        }

        while (totalDone < totalToSubmit) 
        {
            while (totalSubmitted < totalToSubmit) {
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                    }
                    else {
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
                
            }
            if(pq_decode){
                full_diskRequests->pq_callback();
                pq_decode = false;
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncRequest_Full_Batch* req = reinterpret_cast<AsyncRequest_Full_Batch*>((events[i].data));
                if (nullptr != req)
                {   
                    AsyncRequest_Full* single_list = req->request_full.data();
                    for(int j = 0; j < req->list_num; j++)
                    {
                        full_diskRequests->cal_callback(single_list + j, (single_list + j)->m_buffer);
                    }   
                }
            }
            totalQueued = totalDone;

            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                done += _d;
                totalDone += _d;
            }
        }
        
        for (int i = totalQueued; i < totalDone; i++) {

            AsyncRequest_Full_Batch* req = reinterpret_cast<AsyncRequest_Full_Batch*>((events[i].data));
            if (nullptr != req)
            {  
                AsyncRequest_Full* single_list = req->request_full.data();
                for(int j = 0; j < req->list_num; j++)
                {   
                    full_diskRequests->cal_callback(single_list + j, (single_list + j)->m_buffer);
                }   
            }
        }
        full_diskRequests->dp_callback();
    }


    void submit_partially(){
        int iocb_num = partial_diskRequests->list_requests.size();
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;
        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_Partial* readRequest = partial_diskRequests->list_requests.data() + i;
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(partial_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
            iocbs.emplace_back(myiocb);
            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = true;
        int kk = 0;

        if(totalToSubmit == 0){
            //std::cout << "No IO to submit and start PQ\n";
            if(pq_decode){
                partial_diskRequests->pq_callback();
                pq_decode = false;
            }
            return;
        }

        while (totalDone < totalToSubmit) 
        {
            while (totalSubmitted < totalToSubmit) {
                
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                        //std::cout << "Submitted IO: " << submitted  << std::endl;
                    }
                    else {
                        std::cout << "submit fails:" << s << "  submitted:" << submitted << "  total:" << totalToSubmit << std::endl;
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
            }
            
            if(pq_decode){
                //std::cout << "PQ decode called\n";
                partial_diskRequests->pq_callback();
                pq_decode = false;
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncRequest_Partial* req = reinterpret_cast<AsyncRequest_Partial*>((events[i].data));
                if (nullptr != req)
                {   
                    partial_diskRequests->cal_callback(req, req->m_buffer);
                }
            }
            
            totalQueued = totalDone;
            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                if(_d < 0){
                    std::cout << "_d: " << _d << "\n";
                    FAISS_THROW_FMT("To get:%d, error:%s\n", wait, strerror(-_d));
                    exit(1);
                }
                done += _d;
                totalDone += _d;
            }
            //std::cout << "Done IO: " << done  << std::endl;
        }

        
        for (int i = totalQueued; i < totalDone; i++) {
            AsyncRequest_Partial* req = reinterpret_cast<AsyncRequest_Partial*>((events[i].data));
            if (nullptr != req)
            {   
                partial_diskRequests->cal_callback(req, req->m_buffer);
            }
        }
        partial_diskRequests->dp_callback();

    }

    void submit_info(){
        int iocb_num = info_diskRequests->info_requests.size();
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;

        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_IndexInfo* readRequest = info_diskRequests->info_requests.data() + i;
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(info_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
            iocbs.emplace_back(myiocb);
            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = true;
        while (totalDone < totalToSubmit) 
        {
            if (totalSubmitted < totalToSubmit) {
                
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                    }
                    else {
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
            }

            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                done += _d;
                totalDone += _d;
            }
        }
    }


    void submit(int num = -1) override{
        if(num == -1){
            submit_partially();
            partial_diskRequests->page_buffers.clear();

        }else if(num >= 0){
            submit_fully(num);
            full_diskRequests->page_buffers.clear();
        }else if(num == -2){
            submit_info();
        }
        else{
            return;
        }
    }

    int process_page(int* vector_to_submit, int* page_to_search, size_t* vec_page_proj, size_t len_p) override {

        int vector_size = sizeof(ValueType) * d;
        int vec_per_page = PAGE_SIZE/vector_size; 

        int page_num = -1;
        int total_page = 0;
        for(int i = 0; i < len_p; i++){
            int tmp_num = vector_to_submit[i]/vec_per_page;
            vec_page_proj[i] = tmp_num; 
            if(tmp_num != page_num){
                page_to_search[total_page] = tmp_num;
                page_num = tmp_num;
                total_page++;
            }
        }
        return total_page;
    }

    int process_page_transpage_old(int* vector_to_submit, Page_to_Search* page_to_search, size_t* vec_page_proj, size_t len_p){
        int vector_size = sizeof(ValueType) * d;
        for (size_t i = 0; i < len_p; i++) {
            size_t vector_offset = vector_to_submit[i] * vector_size;
            int start_page = vector_offset / PAGE_SIZE;
            int end_page = (vector_offset + vector_size - 1) / PAGE_SIZE; 
            vec_page_proj[i] = start_page;

            page_to_search[i].first = start_page;
            page_to_search[i].last = end_page;
        }
        return len_p;
    }


    int process_page_transpage(int* vector_to_submit, Page_to_Search* page_to_search, size_t* vec_page_proj, size_t len_p)override{
        int vector_size = sizeof(ValueType) * d;
        for (size_t i = 0; i < len_p; i++) {
            size_t vector_offset = vector_to_submit[i] * vector_size;
            int start_page = vector_offset / PAGE_SIZE;
            int end_page = (vector_offset + vector_size - 1) / PAGE_SIZE; 
            vec_page_proj[i] = start_page;

            page_to_search[i].first = start_page;
            page_to_search[i].last = end_page;
        }
        return len_p;
    }


    int get_per_page_element() override {

        return PAGE_SIZE/sizeof(ValueType) ;
    }

    void disk_io_partial_async_pq(AsyncReadRequests_Partial_PQDecode& asyncReadRequests_p) override{
        asyncReadRequests_p.fill_buffer();
        this->partial_diskRequests = &asyncReadRequests_p;
    }

    void disk_io_full_async_pq(AsyncReadRequests_Full_PQDecode& asyncReadRequests_f)override{
        int r_size = asyncReadRequests_f.list_requests.size();
        
        if(r_size!=0){
            asyncReadRequests_f.page_buffers.reserve(r_size);

            for(int i = 0; i < r_size; i++){
                PageBuffer<uint8_t> page_buffer;
                page_buffer.ReservePageBuffer(asyncReadRequests_f.list_requests[i].m_readSize); 
                asyncReadRequests_f.page_buffers.emplace_back(std::move(page_buffer));
            }
            asyncReadRequests_f.fill_buffer(); 
        }
        this->full_diskRequests = &asyncReadRequests_f;
    }

    void disk_io_info_async(AsyncRequests_IndexInfo& asyncReadRequests_i)override{
        int r_size = asyncReadRequests_i.info_requests.size();
        if(r_size!= 0){
            asyncReadRequests_i.fill_buffer();
        }
        this->info_diskRequests = &asyncReadRequests_i;
    }

    void disk_io_all(int D,
                    size_t len,
                    size_t listno, 
                    float* vectors,
                    Aligned_Cluster_Info* acInfo){
         if (fd < 0) {
            std::cerr << "File descriptor is invalid!" << std::endl;
            return;
        }

        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;
        size_t page_count = acInfo[listno].page_count;

        size_t offset = page_start * PAGE_SIZE;
        size_t total_bytes = page_count * PAGE_SIZE;

        PageBuffer<uint8_t> page_buffer;
        page_buffer.ReservePageBuffer(total_bytes); 

        struct iocb cb;
        struct iocb* cbs[1];
        memset(&cb, 0, sizeof(cb));

        cb.aio_fildes = fd;
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_buf = reinterpret_cast<std::uint64_t>(page_buffer.GetBuffer());
        cb.aio_offset = static_cast<std::int64_t>(offset);
        cb.aio_nbytes = total_bytes;
        cb.aio_data = reinterpret_cast<std::uint64_t>(&cb); 

        cbs[0] = &cb;
        int ret = syscall(__NR_io_submit, aio_ctx, 1, cbs);

        struct io_event events[1];
        memset(events, 0, sizeof(events));
        int num_events = 0;
        while(num_events < 1){
            auto _d = syscall(__NR_io_getevents, aio_ctx, 1, 1, events, &AIOTimeout);
            num_events += _d;
        }
        if (reinterpret_cast<uintptr_t>(page_buffer.GetBuffer()) % 512 != 0) {
            std::cerr << "Buffer alignment error for O_DIRECT." << std::endl;
        }
        if (events[0].res != static_cast<size_t>(total_bytes)) {
            std::cerr << "Asynchronous read failed or incomplete!" << std::endl;

        }
        ValueType* data = reinterpret_cast<ValueType*>(page_buffer.GetBuffer());
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < D; ++j) {
                vectors[i * D + j] = static_cast<float>(data[i * D + j]);
            }
        }
    }
};

} // namespace faiss

#endif
