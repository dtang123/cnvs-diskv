// #ifndef FAISS_INDEX_IVFPQFastScanDisk_H
// #define FAISS_INDEX_IVFPQFastScanDisk_H

// #include <memory>
// #include <vector>

// #include <faiss/IndexIVFFastScan.h>
// #include <faiss/IndexIVFPQFastScan.h>
// #include <faiss/impl/ProductQuantizer.h>
// #include <faiss/utils/AlignedTable.h>
// #include <faiss/impl/platform_macros.h>

// #include <string>
// #include <fstream>  

// #include <chrono>
// using namespace std::chrono;

// namespace faiss {
// /* Fast scan version of IVFPQ. Works for 4-bit PQ for now.
//  *
//  * The codes in the inverted lists are not stored sequentially but
//  * grouped in blocks of size bbs. This makes it possible to very quickly
//  * compute distances with SIMD instructions.
//  *
//  * Implementations (implem):
//  * TODO: Range Search
//  * TODO: 0: auto-select implementation (default)
//  * TODO: 1: orig's search, re-implemented
//  * TODO: 2: orig's search, re-ordered by invlist
//  * TODO: 10: optimizer int16 search, collect results in heap, no qbs
//  * TODO: 11: idem, collect results in reservoir
//  * TODO: 12: optimizer int16 search, collect results in heap, uses qbs
//  * TODO: 13: idem, collect results in reservoir
//  */
// struct IndexIVFPQFastScanDisk : public IndexIVFPQFastScan {
//     IndexIVFPQFastScanDisk(
//             Index* quantizer,
//             size_t d,
//             size_t nlist,
//             size_t M,
//             size_t nbits_per_idx,
//             size_t top,
//             float estimate_factor,
//             float prune_factor,
//             const std::string& diskPath,
//             MetricType metric = METRIC_L2);
    
//     IndexIVFPQFastScanDisk();

//     ~IndexIVFPQFastScanDisk();

//     void add(idx_t n, const float* x) override;

//     /*------------------------override from IVFPQFastScan--------------------*/
//     void search_dispatch_implem(
//             idx_t n,
//             const float* x,
//             idx_t k,
//             float* distances,
//             idx_t* labels,
//             const CoarseQuantized& cq,
//             const NormTableScaler* scaler,
//             const IVFSearchParameters* params = nullptr) const override;

//     void search_implem_10(
//             idx_t n,
//             idx_t k,
//             const float* x,
//             SIMDResultHandlerToFloat& handler,
//             const CoarseQuantized& cq,
//             size_t* ndis_out,
//             size_t* nlist_out,
//             //float* local_distances,
//             //idx_t* local_ids,
//             float* distances,
//             idx_t* ids,
//             const NormTableScaler* scaler) const;

//     void set_disk_read(const std::string& diskPath); // Method to set the disk path and
//                                           // open read stream

//     void set_disk_write(const std::string& diskPath); // Method to set the disk path and
//                                           // open write stream

//     void initial_location(idx_t n, const float* data); // Method to initialize the location arrays.  Maybe
//                              // can put it in some other functions. But I'd
//                              // prefer to explicitly calling it now. Must call it before search.
//     void reorganize_vectors(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);

//     void load_from_offset(
//             size_t list_no,
//             size_t offset,
//             float* original_vector);

//     void load_clusters(size_t list_no, float* original_vectors);

//     size_t get_cluster_location(size_t key) const {
//         /*
//         assert()?
//         */
//         return clusters[key];
//     }
//     size_t get_cluster_len(size_t key) const {
//         /*
//         assert()?
//         */
//         return len[key];
//     }
//     size_t get_vector_offset() const {
//         return this->disk_vector_offset;
//     }

//     int get_dim() const {
//         return this->d;
//     }

//     size_t get_top() {
//         return this->top;
//     }

//     void set_top(size_t top) {
//         this->top = top;
//     }

//     float get_estimate_factor() const {
//         return this->estimate_factor;
//     }

//     void set_estimate_factor(float factor) {
//         this->estimate_factor = factor;
//     }

//     float get_estimate_factor_partial() const {
//         return this->estimate_factor_partial;
//     }

//     void set_estimate_factor_partial(float factor) {
//         this->estimate_factor_partial = factor;
//     }

//     float get_prune_factor() const {
//         return this->prune_factor;
//     }

//     void set_prune_factor(float prune_factor) {
//         this->prune_factor = prune_factor;
//     }
    
//     // 1. array to help locate where is the vector in disk. (File is reorganized
//     // by clusters)
//     size_t* clusters; // ith cluster begin at clusters[i]
//     size_t* len;      // with length of len[i]
//     size_t top;
//     float estimate_factor;
//     float estimate_factor_partial;
//     float prune_factor;
//     size_t disk_vector_offset;

//     // 2. disk operations
//     std::string disk_path;
//     std::ifstream disk_data_read;
//     std::ofstream disk_data_write;
// };

// struct IndexIVFPQFastScanDiskStats {
//     size_t full_cluster_compare;
//     ///< compare times in FULL load strategy
//     size_t partial_cluster_compare;
//     ///< compare times in PARTIAL load strategy

//     size_t full_cluster_rerank;
//     ///< rerank times in FULL load strategy
//     size_t partial_cluster_rerank;
//     ///< rerank times in PARTIAL load strategy

//     size_t pruned;

//     std::chrono::duration<double, std::micro> memory_pq_elapsed;
//     std::chrono::duration<double, std::micro> memory_2_elapsed;
//     std::chrono::duration<double, std::micro> disk_full_elapsed;
//     std::chrono::duration<double, std::micro> disk_partial_elapsed;
//     std::chrono::duration<double, std::micro> others_elapsed;
//     std::chrono::duration<double, std::micro> coarse_elapsed;
//     std::chrono::duration<double, std::micro> rank_elapsed;

//     size_t nq;
//     size_t ndis;
//     size_t nlist;



//     IndexIVFPQFastScanDiskStats() {
//         reset();
//     }
//     void reset();
// };

// // global var that collects them all
// FAISS_API extern IndexIVFPQFastScanDiskStats indexIVFPQFastScanDisk_stats;


// } // namespace faiss
// #endif