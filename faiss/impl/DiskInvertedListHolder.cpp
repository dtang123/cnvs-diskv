#include <iostream>
#include <fstream>
#include <stdexcept>
#include <faiss/impl/DiskInvertedListHolder.h>

namespace faiss{

    DiskHolder::DiskHolder(){
        this->aligned_cluster_info = nullptr;
        this->code_size = 0;
        this->disk_path = "";
        this->nlist = 0;
    }

    DiskHolder::DiskHolder(std::string& path,
                            size_t nlist,
                            size_t code_size, 
                            Aligned_Cluster_Info* aligned_cluster_info)
    :disk_path(path), nlist(nlist), code_size(code_size), aligned_cluster_info(aligned_cluster_info),cached_vector(0){

    }

    void DiskHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info){
    
        FAISS_THROW_MSG("DiskHolder::set_holder() is base function.");
    }

    void DiskHolder::warm_up(const std::vector<uint64_t>& cluster_indices){
        FAISS_THROW_MSG("DiskHolder::warm_up() is base function.");
    }

    const unsigned char* DiskHolder::get_cache_data(uint64_t cluster_idx) const{
        FAISS_THROW_MSG("DiskHolder::get_cache_data() is base function.");
    }

    int DiskHolder::is_cached(uint64_t listno, uint64_t pageno = 0) const{
        FAISS_THROW_MSG("DiskHolder::is_cached() is base function.");
    }



    DiskInvertedListHolder::DiskInvertedListHolder(){};

    DiskInvertedListHolder::DiskInvertedListHolder(
                           std::string& path,
                           size_t nlist,
                           size_t code_size,
                           Aligned_Cluster_Info* aligned_cluster_info)
        : DiskHolder(path, nlist, code_size, aligned_cluster_info){
            cached_lists.resize(nlist, -1);
        }
    
    DiskInvertedListHolder::~DiskInvertedListHolder(){

    }

    void DiskInvertedListHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info){

        disk_path = path;                 
        this->nlist = nlist;
        this->code_size = code_size;
        this->aligned_cluster_info = aligned_cluster_info;    
        cached_lists.resize(nlist, -1);   
    }

    void DiskInvertedListHolder::warm_up(const std::vector<uint64_t>& cluster_indices) {
        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }

        for (uint64_t list_no : cluster_indices) {
            if (list_no >= nlist) {
                throw std::out_of_range("Cluster index out of range");
            }

            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;

            size_t cluster_size = acinfo->page_count*PAGE_SIZE;
            size_t cluster_offset = acinfo->padding_offset;
            size_t cluster_begin = acinfo->page_start*PAGE_SIZE;

            this->cached_vector += (cluster_size - cluster_offset)/code_size;

            file.seekg(cluster_begin, std::ios::beg);

            std::vector<unsigned char> buffer(cluster_size);
            file.read(reinterpret_cast<char*>(buffer.data()), cluster_size);

            if (!file) {
                throw std::runtime_error("Error reading cluster from file");
            }

            holder.push_back(std::move(buffer));
            cached_lists[list_no] = holder.size() - 1;
        }

        file.close();
    }

    const unsigned char* DiskInvertedListHolder::get_cache_data(uint64_t cluster_idx) const {
        if (cluster_idx >= holder.size()) {
            throw std::out_of_range("Cluster index out of range");
        }
        return holder[cluster_idx].data();
    }

    DiskVectorHolder::DiskVectorHolder(){}

    DiskVectorHolder::DiskVectorHolder(
        std::string& path,
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info)
        : DiskHolder(path, nlist, code_size, aligned_cluster_info){
    }

    void DiskVectorHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) {
            
            disk_path = path;                 
            this->nlist = nlist;
            this->code_size = code_size;
            this->aligned_cluster_info = aligned_cluster_info; 
            holder.resize(nlist);
            cached_vector_position.resize(nlist);
            cached_lists.resize(nlist, 0);
    }

    void DiskVectorHolder::warm_up(const std::vector<uint64_t>& vector_indices) {

        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }
        std::cout << "Warming up." << std::endl;
        // reserve space
        for(uint64_t lp_no : vector_indices){
            uint64_t list_no = this->lp_listno(lp_no);
            FAISS_ASSERT_MSG(list_no < nlist, "cache vector failed, list_no should smaller than nlist.");
            cached_lists[list_no]++;
        }

        for(int i = 0; i < nlist; i++){
            holder.reserve(code_size * cached_lists[i]);
            cached_vector_position.reserve(cached_lists[i]);
        }

        for (uint64_t lp_no : vector_indices) {

            uint64_t list_no = this->lp_listno(lp_no);
            uint64_t vector_no = this->lp_vectorno(lp_no);

            if (list_no >= nlist) {
                throw std::out_of_range("DiskVectorHolder::warm_up(): Cluster index out of range");
            }

            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;

            size_t cluster_size = acinfo->page_count*PAGE_SIZE;            
            size_t cluster_begin = acinfo->page_start*PAGE_SIZE;

            if(vector_no*code_size > cluster_size){
                throw std::out_of_range("DiskVectorHolder::warm_up(): Vector index out of range");
            }

            size_t vector_disk_begin = cluster_begin + vector_no*code_size;

            this->cached_vector += 1;

            file.seekg(vector_disk_begin, std::ios::beg);

            std::vector<unsigned char> buffer(code_size);
            file.read(reinterpret_cast<char*>(buffer.data()), code_size);

            if (!file) {
                throw std::runtime_error("Error reading cluster from file");
            }

            holder[list_no].insert(holder[list_no].end(), buffer.begin(), buffer.end());
            cached_vector_position[list_no].push_back(vector_no);
        }
        std::cout << "Vecs cached in memory." << std::endl;
        file.close();
    } 

    const unsigned char* DiskVectorHolder::get_cache_data(uint64_t list_no) const {
        return this->holder[list_no].data();
    }

    void DiskVectorHolder_dp::set_holder_dp(
        std::string& path, 
        size_t nlist,
        size_t replicas,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) {
            
            disk_path = path;                 
            this->nlist = nlist;
            this->replicas = replicas;
            this->code_size = code_size;
            this->aligned_cluster_info = aligned_cluster_info; 
    }

    void DiskVectorHolder_dp::warm_up_2(const std::vector<WarmupVecInfo>& vectors_indices) {
        std::unordered_map<idx_t, int> id_to_first_index;
        std::vector<float> id_freq(vectors_indices.size(), 0.0f);
        for (int i = 0; i < vectors_indices.size(); ++i) {
            const auto& item = vectors_indices[i];
            auto it = id_to_first_index.find(item.id);
            if (it == id_to_first_index.end()) {
                id_to_first_index[item.id] = i;
                id_freq[i] = item.freq;
            } else {
                id_freq[it->second] += item.freq;
            }
        }

        local_merge_size = vectors_indices.size();
        min_freq = std::numeric_limits<float>::max();

        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }

        for (int i = 0; i < vectors_indices.size(); ++i) {
            if (id_freq[i] == 0.0f) continue;

            const WarmupVecInfo& item = vectors_indices[i];
            uint64_t lp = item.lp_no;
            size_t list_no = lp >> 32;
            size_t offset = lp & 0xffffffff;

            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;
            size_t vector_disk_offset = acinfo->page_start * PAGE_SIZE + offset * code_size;

            std::vector<uint8_t> buffer(code_size);
            file.seekg(vector_disk_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(buffer.data()), code_size);
            if (!file) {
                throw std::runtime_error("Error reading vector from file");
            }

            CacheNode* node = new CacheNode(item.id, code_size, buffer.data());
            node->freq = id_freq[i]; 
            min_freq = std::min(min_freq, id_freq[i]);

            insert_to_head(node);
            cache_map[item.id] = node;
            ++size;
        }

        file.close();
    }


    bool DiskVectorHolder_dp::insert_cache(idx_t id, const uint8_t* vector_data, float freq) {
        auto it = cache_map.find(id);
        if (it != cache_map.end()) {
            move_to_head(it->second);
            return false;
        }
        if (size >= capacity) {
            evict_tail();
        }
        auto* node = new CacheNode(id, code_size, vector_data, freq);
        insert_to_head(node);
        cache_map[id] = node;
        ++size;
        if (freq < min_freq) min_freq = freq;
        return true;
    }



    std::vector<WarmupVecInfo> DiskVectorHolder_dp::sort_vectors_to_cache(
        const std::vector<std::vector<VectorPair>>& vec_freq,
        size_t nvec
    ) {
        struct FullEntry {
            size_t list_no;
            size_t vecid;
            size_t freq;
            idx_t id;
            bool operator<(const FullEntry& other) const {
                return freq > other.freq;
            }
        };
    
        std::vector<FullEntry> all_entries;

        for (size_t list_no = 0; list_no < vec_freq.size(); list_no++) {
            const auto& vec_list = vec_freq[list_no];
            for (const auto& vp : vec_list) {
                all_entries.push_back({list_no, vp.vecid, vp.freq, vp.id});
            }
        }
    
        if (nvec > all_entries.size()) {
            nvec = all_entries.size();
        }

        std::nth_element(all_entries.begin(), all_entries.begin() + nvec, all_entries.end());
        std::sort(all_entries.begin(), all_entries.begin() + nvec);
    
        std::vector<WarmupVecInfo> result;
        result.reserve(nvec);
    
        for (size_t i = 0; i < nvec; i++) {
            result.emplace_back(lp_build(all_entries[i].list_no, all_entries[i].vecid), all_entries[i].id, ((float)all_entries[i].freq)/nvec);
        }
    
        return result;
    }

    std::vector<size_t> DiskVectorHolder_dp::get_list_offsets(size_t list_no) {
        FAISS_THROW_MSG("Don't ues get_list_offsets function");
        
    }

    uint8_t* DiskVectorHolder_dp::get_cached_data(idx_t id) {
        std::shared_lock<std::shared_mutex> lock(global_mutex);
        auto it = cache_map.find(id);
        if (it != cache_map.end()) { 
            return it->second->data.data(); 
        }
        return nullptr;
    }

    void DiskVectorHolder_dp::merge_from_local(DiskVectorHolder_local* local){
        std::unique_lock<std::shared_mutex> lock(global_mutex);

        size_t merged_vector = 0;
        for (auto& [id, node_local] : local->cache_map) {
            float local_freq = node_local->freq;
            float new_freq = local_freq / (local_merge_size*1.2);
            auto it = cache_map.find(id);

            if (it == cache_map.end()) {
                if (new_freq >= min_freq) {
                    insert_cache(id, node_local->data.data(), new_freq);
                    merged_vector++;
                }
            } else {
                CacheNode* dp_node = it->second;
                float merged_freq = (dp_node->freq * local_merge_size + local_freq) / local_merge_size;
                if (merged_freq > dp_node->freq) {
                    dp_node->freq = merged_freq;
                    move_to_head(dp_node);
                    merged_vector++;
                }
            }
        }
    }
   
    std::vector<WarmupVecInfo> DiskVectorHolder_shard::sort_vectors_to_cache(
        const std::vector<std::vector<VectorPair>>& vec_freq,
        size_t nvec
    ){
        struct FullEntry {
        size_t list_no;
        size_t vecid;
        size_t freq;
        idx_t id;

        bool operator<(const FullEntry& other) const {
            return freq > other.freq;
            }
        };

        std::vector<FullEntry> all_entries;

        for (size_t list_no = 0; list_no < vec_freq.size(); list_no++) {
            const auto& vec_list = vec_freq[list_no];
            for (const auto& vp : vec_list) {
                all_entries.push_back({list_no, vp.vecid, vp.freq, vp.id});
            }
        }

        std::unordered_map<idx_t, size_t> id_first_index;
        std::vector<bool> to_remove(all_entries.size(), false);

        for (size_t i = 0; i < all_entries.size(); ++i) {
            idx_t id = all_entries[i].id;
            auto it = id_first_index.find(id);
            if (it == id_first_index.end()) {
                id_first_index[id] = i;
            } else {
                all_entries[it->second].freq += all_entries[i].freq;
                to_remove[i] = true; 
            }
        }

        std::vector<FullEntry> unique_entries;
        unique_entries.reserve(all_entries.size());
        for (size_t i = 0; i < all_entries.size(); ++i) {
            if (!to_remove[i]) {
                unique_entries.push_back(std::move(all_entries[i]));
            }
        }
        all_entries = std::move(unique_entries); 

        if (nvec > all_entries.size()) {
            nvec = all_entries.size();
        }

        std::nth_element(all_entries.begin(), all_entries.begin() + nvec, all_entries.end());
        std::sort(all_entries.begin(), all_entries.begin() + nvec);

        std::vector<WarmupVecInfo> result;
        result.reserve(nvec);

        for (size_t i = 0; i < nvec; i++) {
            result.emplace_back(
                lp_build(all_entries[i].list_no, all_entries[i].vecid),
                all_entries[i].id,
                static_cast<float>(all_entries[i].freq) / nvec
            );
        }

        return result;
    }

    void DiskVectorHolder_shard::warm_up_2(const std::vector<WarmupVecInfo>& vectors_indices){
        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }

        std::vector<idx_t> ids;
        std::vector<const uint8_t*> vec_ptrs;
        std::vector<std::unique_ptr<uint8_t[]>> vec_data_holder;
        size_t batch_size = 100000;
        size_t vec_acc = 0;

        for (const auto& item : vectors_indices) {
            uint64_t lp = item.lp_no;
            size_t list_no = lp >> 32;
            size_t offset = lp & 0xffffffff;

            
            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;
            size_t vector_disk_offset = acinfo->page_start * PAGE_SIZE + offset * code_size;

            std::unique_ptr<uint8_t[]> buffer(new uint8_t[code_size]);
            file.seekg(vector_disk_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(buffer.get()), code_size);
            if (!file) {
                throw std::runtime_error("Error reading vector from file");
            }

            ids.push_back(item.id);
            vec_ptrs.push_back(buffer.get());
            vec_data_holder.push_back(std::move(buffer));
            vec_acc++;

            if (vec_acc == batch_size) {
                bulk_initialize(ids, vec_ptrs);
                ids.clear();
                vec_ptrs.clear();
                vec_data_holder.clear();
                vec_acc = 0;
            }
        }
        if (vec_acc > 0) {
            bulk_initialize(ids, vec_ptrs);
        }
        file.close();
    }

}