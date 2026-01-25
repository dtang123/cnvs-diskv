#ifndef FAISS_INVLIST_HOLDER_H
#define FAISS_INVLIST_HOLDER_H

#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <mutex>
#include <shared_mutex>
#include <iostream>
#include <cstring>
#include <list>
#include <atomic>
#include <memory>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/DiskIOStructure.h>
#include <faiss/MetricType.h>


namespace faiss {

enum CACHE_UPDATE_STRATEGY{
    NO_UPDATE = 0,              // Default, no update cache, just use static cache
    IMMEDIATELY_UPDATE = 1,     // Update cache immediately after I/O
    SINGLE_THREAD_UPDATE = 2,   // Update cache in every single thread (every 50 queries in a thread)
    GLOBAL_CONTROL_UPDATE = 3,  // Update cache only when satisfy the global condition (every 500 queries)
    SINGLE_THREAD_LOCAL_BUFFER_UPDATE = 4, // Update cache in every single thread, but use local buffer to store the vectors
};



struct DiskHolder{
    std::string disk_path;
    size_t code_size;           // code_size is calculated by Bytes.  eg 128D float--> 128*4B,  128D uint8_t--> 128B
    size_t nlist;
    size_t cached_vector;
    Aligned_Cluster_Info* aligned_cluster_info;

    DiskHolder();

    DiskHolder(std::string& path,
                size_t nlist,
                size_t code_size,
                Aligned_Cluster_Info* aligned_cluster_info);
    
    virtual void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info);

    virtual void warm_up(const std::vector<uint64_t>& cluster_indices); 

    virtual const unsigned char* get_cache_data(uint64_t cluster_idx) const;

    virtual int is_cached(uint64_t listno, uint64_t pageno) const;

    //used in warm_up() so that we can override with listno << 32 | pageno
    inline uint64_t lp_build(uint64_t list_id, uint64_t offset) {
        return list_id << 32 | offset;
    }

    inline uint64_t lp_listno(uint64_t lo) {
        return lo >> 32;
    }

    inline uint64_t lp_vectorno(uint64_t lo) {
        return lo & 0xffffffff;
    }
};



struct DiskInvertedListHolder : DiskHolder {
    std::vector<std::vector<unsigned char>> holder; // Vector to hold clusters data in memory
    std::vector<int> cached_lists;
    
    DiskInvertedListHolder();

    DiskInvertedListHolder(std::string& path,
                           size_t nlist,
                           size_t code_size,
                           Aligned_Cluster_Info* aligned_cluster_info);
    
    ~DiskInvertedListHolder();

    void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) override;

    void warm_up(const std::vector<uint64_t>& cluster_indices) override; 

    const unsigned char* get_cache_data(uint64_t cluster_idx) const override;

    int is_cached(uint64_t listno, uint64_t page_no = 0) const override{
        return cached_lists[listno];
    }
};


struct VectorCollector{

    VectorCollector(){

    }

    VectorCollector(int nlist){
        vector_collector.resize(nlist);
    }

    std::vector<std::vector<size_t>> vector_collector;
};


struct VectorPair{
    size_t freq = 0;
    size_t vecid;  
    idx_t id;
};




struct DiskVectorHolder : DiskHolder{
    
    // holder has nlist std::vector<unsigned char>, each stores the vectors cached.
    // cached_vector_position record the i th cached vectors position in the cluster.
    // std::vector<> vecs;
    std::vector<std::vector<unsigned char>> holder; // Vector to hold clusters data in memory
    std::vector<std::vector<int>> cached_vector_position;

    bool expand_candidate = false;

    // lists that have vectors cached
    std::vector<int> cached_lists;

    DiskVectorHolder();

    DiskVectorHolder(std::string& path,
                    size_t nlist,
                    size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
                    Aligned_Cluster_Info* aligned_cluster_info);

    void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) override;

    // Warm up the specified vectors by reading them into memory
    // Vector indices =  list_no (32)  + in list offset(32)
    void warm_up(const std::vector<uint64_t>& vectors_indices) override; 
    
    // use this position to compare whether a vector is cached.
    const int* get_cached_vector_position(int list_no) const{
        if(cached_vector_position[list_no].empty()){
            return nullptr;
        }
        else{
            return cached_vector_position[list_no].data();
        }
    }

    // Access cluster data from memory
    // it doesn't return single vector, it will return all the vectors cached in a list. 
    const unsigned char* get_cache_data(uint64_t vector_idx) const override;

    // Judge whether a list has cached vector
    int is_cached(uint64_t listno, uint64_t page_no = 0) const override{
        
        return cached_lists.empty()? 0 : cached_lists[listno];
    }

    std::vector<size_t> sort_vectors_to_cache(
        const std::vector<std::vector<VectorPair>>& vec_freq,
        size_t nvec
    ) {
        struct FullEntry {
            size_t list_no;
            size_t vecid;
            size_t freq;
    
            bool operator<(const FullEntry& other) const {
                return freq > other.freq;
            }
        };
    
        std::vector<FullEntry> all_entries;
    
        for (size_t list_no = 0; list_no < vec_freq.size(); list_no++) {
            const auto& vec_list = vec_freq[list_no];
            for (const auto& vp : vec_list) {
                all_entries.push_back({list_no, vp.vecid, vp.freq});
            }
        }
        if (nvec > all_entries.size()) {
            nvec = all_entries.size();
        }
    
        std::nth_element(all_entries.begin(), all_entries.begin() + nvec, all_entries.end());
        std::sort(all_entries.begin(), all_entries.begin() + nvec);
    
        std::vector<size_t> result;
        result.reserve(nvec);
    
        for (size_t i = 0; i < nvec; i++) {

            std::cout << "listno:" << all_entries[i].list_no << "  vecid:" << all_entries[i].vecid << "  freq:" << all_entries[i].freq << std::endl;

            result.push_back(lp_build(all_entries[i].list_no, all_entries[i].vecid));
        }
    
        return result;
    }

};


struct DiskVectorHolder_local;

struct WarmupVecInfo {
    uint64_t lp_no;
    idx_t id;
    float freq;
    WarmupVecInfo(uint64_t lp, idx_t id_, float freq_= 0) : lp_no(lp), id(id_) ,freq(freq_){}
};

struct CacheNode {
    idx_t id;
    std::vector<uint8_t> data;
    CacheNode* prev = nullptr;
    CacheNode* next = nullptr;

    float freq = 0;

    CacheNode(idx_t id_, size_t code_size, const uint8_t* src, float freq_ = 0)
        : id(id_), data(code_size), freq(freq_) {
        std::memcpy(data.data(), src, code_size);
    }
};

struct DiskVectorHolder_dp : DiskHolder{
    // Structure 1: hash table storing deduplicated vector IDs and std::vector<list_no>
    // Structure 2: linked list that holds the actual vectors
    // Structure 3: two-level linked list tracking vector positions per list

    // Structures 1 and 2 function as an LRU for recently used vectors
    // This holder tracks the current cache size and the maximum capacity
    // Warm-up preloads some cached vectors
    // insert_cache() adds more vectors afterwards

    // Replica count defaults to one
    int replicas = 1;

    size_t capacity;
    size_t size;

    float min_freq;
    size_t local_merge_size = 500;

    std::unordered_map<idx_t, CacheNode*> cache_map;
    CacheNode* head;
    CacheNode* tail;
    std::shared_mutex global_mutex;

    //std::vector<ListNode*> list_heads;
    //std::vector<std::mutex> list_locks;


    DiskVectorHolder_dp(size_t capacity_, size_t code_size_, size_t nlist_, std::string& disk_path_, Aligned_Cluster_Info* aligned_cluster_info_)
        :DiskHolder(disk_path_, nlist_, code_size_, aligned_cluster_info_), capacity(capacity_), size(0), head(nullptr), tail(nullptr) {
        //list_heads.resize(nlist, nullptr);
        //list_locks = std::vector<std::mutex>(nlist);
    }
    
    void set_holder_dp(std::string& path, 
                    size_t nlist,
                    size_t replicas,
                    size_t code_size,
                    Aligned_Cluster_Info* aligned_cluster_info);
    
    // Warm up the specified vectors by reading them into memory
    // Vector indices =  list_no (32)  + in list offset(32)
    // Before warm up, we need to make sure there is no replicas
    void warm_up_2(const std::vector<WarmupVecInfo>& vectors_indices); 
    
    // Use structure 3 to detect whether a list is cached; return the representative chain when present
    std::vector<size_t> get_list_offsets(size_t list_no);

    uint8_t* get_cached_data(idx_t id);
    
    // The function receives a vector id and payload; first call is_vec_cached to check its presence in the hash
    // If cached, simply update structure 3 with the list entry; otherwise insert into the hash and update structure 3 accordingly
    // Because of the LRU policy, evict the least recently used vector when the cache is full—removing it from structures 1 and 2
    // The hash also stores std::vector<list_no>, so ensure structure 3 is updated in reverse
    // vector_data size depends on code_size
    // BUG: this path is no longer usable

    //bool insert_cache(idx_t id, size_t list_no, size_t offset_in_list, const uint8_t* vector_data);

    
    // Sort by frequency of occurrence
    std::vector<WarmupVecInfo> sort_vectors_to_cache(const std::vector<std::vector<VectorPair>>& vec_freq, size_t nvec); 

    void merge_from_local(DiskVectorHolder_local* local);

//private:

    bool insert_cache(idx_t id, const uint8_t* vector_data, float freq);
    
    void insert_to_head(CacheNode* node) {
        node->next = head;
        node->prev = nullptr;
        if (head) head->prev = node;
        head = node;
        if (!tail) tail = node;
    }

    void move_to_head(CacheNode* node) {
        if (node == head) return;
        if (node->prev) node->prev->next = node->next;
        if (node->next) node->next->prev = node->prev;
        if (node == tail) tail = node->prev;
        node->next = head;
        node->prev = nullptr;
        if (head) head->prev = node;
        head = node;
    }

    void evict_tail() {
        if (!tail) return;
        CacheNode* node = tail;
        if (tail->prev) {
            tail = tail->prev;
            tail->next = nullptr;
        } else {
            head = tail = nullptr;
        }
        cache_map.erase(node->id);
        delete node;
        --size;
    }

    void evict_n_tails(size_t n) {
        for (size_t i = 0; i < n && tail; ++i) {
            CacheNode* node = tail;
            if (tail->prev) {
                tail = tail->prev;
                tail->next = nullptr;
            } else {
                head = tail = nullptr;
            }
            cache_map.erase(node->id);
            delete node;
            --size;
        }
    }
    
};



struct DiskVectorHolder_local {
    size_t capacity = 0;
    size_t size = 0;
    std::unordered_map<idx_t, CacheNode*> cache_map;
    CacheNode* head = nullptr;
    CacheNode* tail = nullptr;

    std::atomic<bool> need_merge = false;
    std::shared_mutex local_mutex;

    DiskVectorHolder_local(size_t capacity_)
    :capacity(capacity_){

    }

    virtual uint8_t* get_cached_data(idx_t id) {
        auto it = cache_map.find(id);
        if (it != cache_map.end()) {
            return it->second->data.data();
        }
        return nullptr;
    }

    virtual bool insert_cache(idx_t id, const uint8_t* vector_data, size_t code_size) {
        auto it = cache_map.find(id);
        if (it != cache_map.end()) {
            it->second->freq++;
            return false;
        } 

        if (size >= capacity) {
            evict_tail();
        }

        auto* node = new CacheNode(id, code_size, vector_data, 1);
        node->next = head;
        if (head) head->prev = node;
        head = node;
        if (!tail) tail = node;
        cache_map[id] = node;
        ++size;
        return true;
    }

    virtual void evict_tail() {
        if (!tail) return;
        CacheNode* node = tail;

        if (tail->prev) {
            tail = tail->prev;
            tail->next = nullptr;
        } else {
            head = tail = nullptr;
        }

        cache_map.erase(node->id);
        delete node;
        --size;
    }

    virtual void clear() {
        CacheNode* node = head;
        while (node) {
            CacheNode* next = node->next;
            delete node;
            node = next;
        }
        cache_map.clear();
        head = tail = nullptr;
        size = 0;
    }

    virtual ~DiskVectorHolder_local() = default;
};


struct DiskVectorHolder_local_buffer : DiskVectorHolder_local{
    
    std::vector<uint8_t> cache_buffer; 
    std::vector<idx_t> cache_id;  
    size_t code_size;
    size_t current_position = 0;
    bool check_dup = false;
    std::unordered_map<idx_t, size_t> id_to_pos; 

    DiskVectorHolder_local_buffer(size_t capacity_, size_t code_size_, bool check_dup_ = false)
        : DiskVectorHolder_local(capacity_), code_size(code_size_), check_dup(check_dup_) {
        cache_buffer.resize(capacity * code_size);
        cache_id.resize(capacity, static_cast<idx_t>(-1)); 
    }

    uint8_t* get_cached_data(idx_t id) override {
        if (check_dup) {
            auto it = id_to_pos.find(id);
            if (it != id_to_pos.end()) {
                return &cache_buffer[it->second * code_size];
            }
        } else {
            for (size_t i = 0; i < capacity; ++i) {
                if (cache_id[i] == id) {
                    return &cache_buffer[i * code_size];
                }
            }
        }
        return nullptr;
    }

    bool insert_cache(idx_t id, const uint8_t* vector_data, size_t code_size_) override {
        if (check_dup) {
            if (id_to_pos.count(id)) return false;
        }
        uint8_t* dest = &cache_buffer[current_position * code_size];
        std::memcpy(dest, vector_data, code_size);
        if (check_dup) {
            if (cache_id[current_position] != static_cast<idx_t>(-1)) {
                id_to_pos.erase(cache_id[current_position]);
            }
            id_to_pos[id] = current_position;
        }

        cache_id[current_position] = id;
        current_position++;

        if (current_position >= capacity) {
            current_position = 0;
            need_merge.store(true, std::memory_order_relaxed);
        }

        return true;
    }

    void clear() override {
        std::fill(cache_id.begin(), cache_id.end(), static_cast<idx_t>(-1));
        if (check_dup) id_to_pos.clear();
        current_position = 0;
        need_merge.store(false, std::memory_order_relaxed);
    }

};




struct DiskVectorHolder_shard: DiskHolder{
public:

    struct Node {
        idx_t id;
        std::vector<uint8_t> data;

        Node(idx_t id_, const uint8_t* src, size_t code_size)
            : id(id_), data(src, src + code_size) {}
    };
    struct Shard {
        std::list<Node> lru_list;
        std::unordered_map<idx_t, std::list<Node>::iterator> map;
        mutable std::shared_mutex mutex;

        void insert_no_limit(idx_t id, const uint8_t* vec_data, size_t code_size) {
            auto it = map.find(id);
            if (it != map.end()) {
                //it->second->data.assign(vec_data, vec_data +code_size);
                lru_list.splice(lru_list.begin(), lru_list, it->second);
                return;
            }

            lru_list.emplace_front(id, vec_data, code_size);
            map[id] = lru_list.begin();
        }

        bool insert_with_eviction(idx_t id, const uint8_t* vec_data, size_t code_size, std::atomic<size_t>& global_size, size_t global_capacity) {
            std::unique_lock lock(mutex);

            auto it = map.find(id);
            if (it != map.end()) {
                //it->second->data.assign(vec_data, vec_data + code_size);
                lru_list.splice(lru_list.begin(), lru_list, it->second);
                return false;
            }
            if (global_size.load(std::memory_order_relaxed) >= global_capacity && !lru_list.empty()) {
                auto last = --lru_list.end();
                map.erase(last->id);
                lru_list.pop_back();
                global_size--;
            }

            lru_list.emplace_front(id, vec_data, code_size);
            map[id] = lru_list.begin();
            global_size++;
            return true;
        }

        const uint8_t* get(idx_t id) {
            std::unique_lock lock(mutex);
            auto it = map.find(id);
            if (it == map.end()) return nullptr;
            lru_list.splice(lru_list.begin(), lru_list, it->second); // move to front
            return it->second->data.data();
        }

        void clear(std::atomic<size_t>& global_size) {
            std::unique_lock lock(mutex);
            global_size -= map.size();
            map.clear();
            lru_list.clear();
        }
    };

    DiskVectorHolder_shard(size_t n_shards_, size_t code_size_, size_t global_capacity_,size_t nlist_, std::string& disk_path_, Aligned_Cluster_Info* aligned_cluster_info_)
        :DiskHolder(disk_path_, nlist_, code_size_, aligned_cluster_info_), n_shards(n_shards_),  global_capacity(global_capacity_), shards(n_shards_), global_size(0) {
        for (auto& shard : shards) {
            shard = std::make_unique<Shard>();
        }
    }

    void insert(idx_t id, const uint8_t* vec_data) {
        get_shard(id)->insert_with_eviction(id, vec_data, code_size, global_size, global_capacity);
    }

    void insert_batch(DiskVectorHolder_local* local){
        size_t merged_vector = 0;
        for (auto& [id, node_local] : local->cache_map) {
            insert(id, node_local->data.data());
            merged_vector++;
        }
    }

    void insert_batch_buffer(DiskVectorHolder_local* local){
        auto* buffer_local = dynamic_cast<DiskVectorHolder_local_buffer*>(local);
        if (!buffer_local) return; 

        size_t merged_vector = 0;
        size_t code_size = buffer_local->code_size;

        for (size_t i = 0; i < buffer_local->capacity; ++i) {
            idx_t id = buffer_local->cache_id[i];
            if (id == static_cast<idx_t>(-1)) continue;
            const uint8_t* data_ptr = &buffer_local->cache_buffer[i * code_size];
            insert(id, data_ptr); 
            merged_vector++;
        }

        // optional
        buffer_local->clear();
    }


    const uint8_t* get(idx_t id) {
        return get_shard(id)->get(id);
    }

    void erase(idx_t id) {
        auto shard = get_shard(id);
        std::unique_lock lock(shard->mutex);
        auto it = shard->map.find(id);
        if (it != shard->map.end()) {
            shard->lru_list.erase(it->second);
            shard->map.erase(it);
            global_size--;
        }
    }

    void clear() {
        for (auto& shard : shards) {
            shard->clear(global_size);
        }
    }

    void bulk_initialize(const std::vector<idx_t>& ids, const std::vector<const uint8_t*>& vecs) {
        if (ids.size() != vecs.size()) {
            throw std::runtime_error("ids and vecs must have the same length");
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            if(vecs[i] == nullptr){
                std::cout <<"NULLPTR!!!\n";
                exit(10); 
            }
            get_shard(ids[i])->insert_no_limit(ids[i], vecs[i], code_size);
            global_size++;
        }
    }

    size_t get_global_size() const {
        return global_size.load();
    }


    std::vector<WarmupVecInfo> sort_vectors_to_cache(
        const std::vector<std::vector<VectorPair>>& vec_freq,
        size_t nvec
    );
    void warm_up_2(const std::vector<WarmupVecInfo>& vectors_indices); 


private:
    size_t n_shards;
    //size_t code_size;
    size_t global_capacity;
    std::atomic<size_t> global_size;
    std::vector<std::unique_ptr<Shard>> shards;

    Shard* get_shard(idx_t id) {
        return shards[id % n_shards].get();
    }

};


} // namespace faiss

#endif
