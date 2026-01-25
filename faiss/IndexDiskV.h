#ifndef FAISS_INDEX_DISKV_H
#define FAISS_INDEX_DISKV_H

#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/impl/DiskInvertedListHolder.h>
#include <faiss/impl/DiskIOProcessor.h>

#include <string>
#include <fstream>   

#include <chrono>

#define CACHE_MODE

namespace faiss {

    //struct Global_DiskResultHandler;
namespace{
    struct DiskResultHandler;
    struct UncachedList;
    typedef std::vector<UncachedList> UncachedLists;

}


class IndexDiskV : public IndexIVFPQ {
public:

    IndexDiskV(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            size_t top,
            float estimate_factor,
            float prune_factor,
            const std::string& diskPath,
            const std::string& valueType = "float",
            MetricType metric = METRIC_L2);
    
    IndexDiskV();

    ~IndexDiskV();


    void set_disk_read(const std::string& diskPath); // Method to set the disk path and open read stream

    void set_disk_write(const std::string& diskPath);  // Method to set the disk path and open write stream

    void initial_location(idx_t n, const float* data); // Method to initialize the location arrays.  Maybe
                             // can put it in some other functions. But I'd
                             // prefer to explicitly calling it now. Must call it before search.
    void reorganize_vectors(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);
    void reorganize_vectors_2(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);

    void load_from_offset(
            size_t list_no,
            size_t offset,
            float* original_vector);

    void load_clusters(size_t list_no, float* original_vectors);

    size_t get_cluster_location(size_t key) const {
        /*
        assert()?
        */
        return clusters[key];
    }
    size_t get_cluster_len(size_t key) const {
        /*
        assert()?
        */
        return len[key];
    }
    size_t get_vector_offset() const {
        return this->disk_vector_offset;
    }

    const std::string& get_disk_path() const {
        return this->disk_path;
    }
    
    int get_dim() const {
        return this->d;
    }

    size_t get_top() {
        return this->top;
    }

    void set_top(size_t top) {
        this->top = top;
    }

    float get_estimate_factor() const {
        return this->estimate_factor;
    }

    void set_estimate_factor(float factor) {
        this->estimate_factor = factor;
    }

    
    float get_estimate_factor_partial() const {
        return this->estimate_factor_partial;
    }

     // manually set it, or it is same with estimate_factor
    void set_estimate_factor_partial(float factor) {
        this->estimate_factor_partial = factor;
    }

    void set_estimate_factor_high_dim(float factor) {
        this->estimate_factor_high_dim = factor;
    }


    float get_prune_factor() const {
        return this->prune_factor;
    }

    void set_prune_factor(float prune_factor) {
        this->prune_factor = prune_factor;
    }

    void set_centroid_index_path(const std::string& centroid_path) {
        centroid_index_path = centroid_path;
    }

    void train_graph() override;

    void load_hnsw_centroid_index();

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void in_list_clustered_id(
        idx_t n, 
        const float* x, 
        const faiss::idx_t* coarse_idsx,
        idx_t begin_id, 
        idx_t* clustered_xids);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    
    //void train(idx_t n, const float* x) override;


    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in = nullptr ) const override;

    void search_fully(
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
            ) const;


    void search_partially(
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
            ) const;

    void search_o(
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
        ) const;
    
    void search_uncached(
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
            ) const;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* ivf_stats) const override;

    DiskIOProcessor* get_DiskIOBuildProcessor();

    DiskIOProcessor* get_DiskIOSearchProcessor() const;

    void initializeDiskIO(int n_threads);
    void shutdownDiskIO(int n_threads);
    // warm up according to vectors
    int warmUpVectorCache(size_t n, float* x, size_t w_nprobe, size_t k,size_t nvec, float efp,size_t n_threads);
    int warmUpVectorCacheDp(size_t n, float* x, size_t w_nprobe, size_t k,size_t nvec, float efp,size_t n_threads);

    int warmUpVectorCacheShard(size_t n, float* x, size_t w_nprobe, size_t k,size_t nvec,float efp, size_t n_threads, size_t n_shard);
    // warm up according to nlist
    int warmUpListCache(size_t n, float* x, size_t w_nprobe, size_t nlist);

    int warmUpPageCache(size_t n, float* x, size_t w_nprobe, size_t npage);

    int warmUpIndexMetadata(size_t n, float* x, size_t w_nprobe, size_t warm_list);

    int merge_cache_local_to_global() const;

    void trigger_local_merge_flag() const;

    void set_cache_strategy(CACHE_UPDATE_STRATEGY strategy, size_t batch_size = 50, bool mode_1 = false);

    void warmUpAllIndexMetadata();

    //? method 1 :rewrite
    std::vector<size_t> search_for_cache(idx_t n, const float* x, size_t nprobe, idx_t k, size_t top);
    //  method 2: add something in search_o
    void initializeVectorCollector(int n_threads);
    std::vector<std::vector<VectorPair>> finalizeVectorCollector(int n_threads);
    

    void initializeLocalVectorCaches(int n_threads, size_t capacity);

    void releaseLocalVectorCaches(int n_threads);

    size_t get_code_size() const{
        if(valueType == "float"){
            return d*sizeof(float);
        }else if(valueType == "uint8"){
            return d*sizeof(uint8_t);
        }else if(valueType == "int8"){
            return d*sizeof(int8_t);
        }else{
            FAISS_THROW_MSG("get_code_size() only support float & uint8_t");
            return 0;
        }
    }

    // store PQ file in the same file with vectors
    int set_select_lists_mode(std::string select_lists_path = ""){
        select_lists = true;
        this->select_lists_path = disk_path;

        cached_list_info = new bool[nlist];

        return 0;
    }


    void set_in_list_clustering(){
        this->in_list_clustering = true;
    }

    void set_soaring(){
        this->soaring = true;
    }

    void set_max_continous_pages(int max_page){
        this->max_continous_pages = max_page;
    }

    void set_ahead_pq_volumes(int full = 1, int partial_one = 1, int partial_two = 1){
        full_decode_volume = full;
        partial_one_decode_volume = partial_one;
        partial_two_decode_volume = partial_two;
    }

    void set_shrink_replica(float shrink_factor){
        this->shrink_replica = true;
        this->shrink_factor = shrink_factor;
    }

    void set_memory_graph_build_params(int ef_construction, int ef_search, int graph_M);

    void set_multi_index(size_t order, size_t partition_num){
        use_multi_ivf = true;
        this->partition_num = partition_num;
        this->order = order;
    }

    void set_submit_per_round(size_t spr){
        this->submit_per_round = spr;
    }   

    void set_search_threshold(size_t nq, std::vector<float> res){
        use_search_threshold = true;
        
        this->search_threshold = new float[nq];
        for(int i = 0; i < nq; i++){
            search_threshold[i] = res[i];

        }
    }

    void clear_search_threshold(){
        if(use_search_threshold){
            use_search_threshold = false;
            if(this->search_threshold != nullptr){
                delete[] this->search_threshold;
                this->search_threshold == nullptr;
            }
        }
    }

    // modify ith vector's index i to i*n + order.
    void multi_ivf_modify_id();

    

    

// Search parameters
    // 1. array to help locate where is the vector in disk. (File is reorganized
    // by clusters)
    size_t* clusters;     // ith cluster begin at clusters[i]
    size_t* len;          // with length of len[i]
    size_t top;
    float estimate_factor;
    float estimate_factor_partial;   // usually same with estimate_factor
    float estimate_factor_high_dim = 1;   // use in very high dim vector
    float prune_factor;              // prune some lists
    size_t disk_vector_offset;

    int max_continous_pages = 1;    // merge parameter
    
    int full_decode_volume = 1;      // PQ decode in advance
    int partial_one_decode_volume= 1;
    int partial_two_decode_volume= 1;
    size_t submit_per_round = 5;    // default


    bool use_search_threshold = false;
    float* search_threshold;
    size_t* q_indicator = nullptr;

    // Only use in Multi_ivfdiskann
    // id will be modified when it's true in building
    // threshold will be affected if it's true in searching
    bool use_multi_ivf = false;
    float* acc_dis;   // distances that be computed by prior indices
    idx_t* acc_ids;

    size_t partition_num = 1;
    size_t order = 0;

    //DiskResultHandler* global_result_handler;


    mutable std::vector<std::unique_ptr<DiskIOProcessor>> diskIOprocessors;

    // Cache
    DiskInvertedListHolder diskInvertedListHolder;
    DiskVectorHolder diskVectorHolder;

    // record nthread the queries had been searched
    // if any one reaches local_merge_size, submit caches of all threads
    mutable std::vector<size_t> result_cached_queries;   
    bool switch_result_cached_queries = false;
    float local_cache_submit_ratio = 3;


    mutable DiskVectorHolder_dp* diskVectorHolder_dp = nullptr;
    mutable std::vector<std::unique_ptr<DiskVectorHolder_local>> diskVectorHolders_local;

    // buffer cache
    bool cache_on_buffer = false;
    size_t cache_buffer_size = 0;
    bool cache_buffer_remove_dup = false;
    



    Aligned_Cluster_Info* aligned_cluster_info;
    
    bool vector_cache_setting_mode = false;
    bool vector_cache_dp_setting_mode = false;
    bool runtime_cache = false;
    size_t local_cache_capacity = 0; 
    size_t local_merge_size = 0;
    mutable std::vector<std::unique_ptr<VectorCollector>> vectorCollectors;

    // cache version 3
    mutable DiskVectorHolder_shard* diskVectorHolder_shard = nullptr;
    bool runtime_shard_update = false;
    bool batch_shard_update = false;
    bool global_shard_update = false;
    mutable std::atomic<size_t> global_query_count = 0;
    size_t batch_shard_query_size = 0;

    CACHE_UPDATE_STRATEGY cache_strategy = IMMEDIATELY_UPDATE;


// Build parameters
    // 1. build all in memory, only support build by float!
    ArrayInvertedLists* build_invlists;

    // 2. disk operations
    size_t add_batch_num = 2;    // add a big file in batchs
    size_t actual_batch_num = 0;  // temperory varible..... delete it later
    bool reorganize_lists = false;

    bool in_list_clustering = false;   // control whether do in list clustering

    bool soaring = false;     // control whether use soar to assign vector

    //instead reading invlist from IndexRead, reading from compress_list 
    bool select_lists = false;
    std::string select_lists_path = "";
    Aligned_Invlist_Info* aligned_inv_info;

    std::string disk_path;
    int build_memory_graph_ef_construction = 40;
    int build_memory_graph_ef_search = 16;
    int build_memory_graph_M = 16;
    //std::string disk_path_clustered;
    std::ifstream disk_data_read;
    std::ofstream disk_data_write;

    // 3. extra graph index
    std::string centroid_index_path;
    faiss::IndexHNSWFlat* centroid_index = nullptr;

    // 4. value type, must set when constructing
    std::string valueType;


    //5. shrink_replica
    bool shrink_replica =false;
    float shrink_factor = 1.15;

    // delete it somewhere
    float* xb;

// Cache parameter
    // cached list info(pq + index + map)
    bool* cached_list_info;

};

struct IndexDiskVStats {

    size_t full_cluster_compare;
    ///< compare times in FULL load strategy
    size_t partial_cluster_compare;
    ///< compare times in PARTIAL load strategy

    size_t full_cluster_rerank;
    ///< rerank times in FULL load strategy
    size_t partial_cluster_rerank;
    ///< rerank times in PARTIAL load strategy

    size_t cached_list_access;
    size_t cached_vector_access;

    size_t pq_list_full;
    size_t pq_list_partial;

    size_t searched_vector_full;
    size_t searched_vector_partial;

    size_t searched_page_full;
    size_t searched_page_partial;

    size_t requests_full;
    size_t requests_partial;

    size_t cached_vectors_partial;

    std::chrono::duration<double, std::micro> memory_1_elapsed;
    std::chrono::duration<double, std::micro> memory_2_elapsed;
    std::chrono::duration<double, std::micro> memory_3_elapsed;
    std::chrono::duration<double, std::micro> disk_full_elapsed;
    std::chrono::duration<double, std::micro> disk_partial_elapsed;
    std::chrono::duration<double, std::micro> others_elapsed;
    std::chrono::duration<double, std::micro> coarse_elapsed;
    std::chrono::duration<double, std::micro> rank_elapsed;  
    std::chrono::duration<double, std::micro> rerank_elapsed;
    std::chrono::duration<double, std::micro> pq_elapsed;
    std::chrono::duration<double, std::micro> cached_calculate_elapsed;
    std::chrono::duration<double, std::micro> delete_elapsed;

    std::chrono::duration<double, std::micro> memory_uncache_elapsed;
    std::chrono::duration<double, std::micro> rank_uncache_elapsed;  
    std::chrono::duration<double, std::micro> disk_uncache_calc_elapsed;
    std::chrono::duration<double, std::micro> disk_uncache_info_elapsed;
    std::chrono::duration<double, std::micro> pq_uncache_elapsed;

    std::chrono::duration<double, std::micro> full_duplicate_elapsed;
    std::chrono::duration<double, std::micro> partial_duplicate_elapsed;

    std::chrono::duration<double, std::micro> cache_system_get_elapsed;
    std::chrono::duration<double, std::micro> cache_system_insert_elapsed;

    size_t searched_lists;
    size_t pruned;

    IndexDiskVStats() {
        reset();
    }
    void reset();
};

// global var that collects them all
FAISS_API extern IndexDiskVStats indexDiskV_stats;

}

#endif
