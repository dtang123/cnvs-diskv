#include "IndexIVFDiskANN.h"

#include <faiss/utils/utils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <iostream>
#include <omp.h>

#include <cinttypes>
#include <cstdio>

#include <string>

#include <DiskANN/include/utils.h>
#include <DiskANN/include/disk_utils.h>
#include <DiskANN/include/math_utils.h>
#include <DiskANN/include/index.h>
#include <DiskANN/include/partition.h>
//#include <DiskANN/include/program_options_utils.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <DiskANN/include/linux_aligned_file_reader.h>

#define WARMUP false

namespace faiss {

IndexIVFDiskANN::~IndexIVFDiskANN(){
    if(this->disk_index_offset!=nullptr){
        delete[] disk_index_offset;
        disk_index_offset = nullptr;
    }
}

IndexIVFDiskANN::IndexIVFDiskANN(){};

IndexIVFDiskANN::IndexIVFDiskANN(
    Index* quantizer,
    size_t d,
    size_t nlist,
    size_t M_pq,
    size_t nbits_per_idx,

    float B,
    float M,
    uint32_t num_threads,
    uint32_t R,
    uint32_t L,
    const std::string& data_type,
    const std::string& dist_fn,
    const std::string& data_path,
    const std::string& index_path_prefix,
    const std::string& codebook_prefix,
    const std::string& label_file,
    const std::string& universal_label,
    const std::string& label_type,
    uint32_t disk_PQ,
    uint32_t build_PQ,
    uint32_t QD,
    uint32_t Lf,
    uint32_t filter_threshold,
    
    bool append_reorder_data,
    bool use_opq,
    MetricType metric)
    : IndexIVFFlat(quantizer, d, nlist, metric),
      data_type(data_type),
      dist_fn(dist_fn),
      data_path(data_path),
      index_path_prefix(index_path_prefix),
      codebook_prefix(codebook_prefix),
      label_file(label_file),
      universal_label(universal_label),
      label_type(label_type),
      num_threads(num_threads),
      R(R),
      L(L),
      disk_PQ(disk_PQ),
      build_PQ(build_PQ),
      QD(QD),
      Lf(Lf),
      filter_threshold(filter_threshold),
      B(B),
      M(M),
      append_reorder_data(append_reorder_data),
      use_opq(use_opq) {
    this->disk_index_offset = new size_t[nlist];
    std::cout << "IndexIVFDiskANN initialized with updated DiskANN parameters.\n";
}


// IVFFlat
void IndexIVFDiskANN::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context)
{
    idx_t num_centroids=assign_replicas;
    std::cout<< "num_centroids: "<<num_centroids<<std::endl;

    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            for (size_t j = 0; j < num_centroids; j++) {
                idx_t list_no = coarse_idx[i * num_centroids + j];
                if (list_no >= 0 && list_no % nt == rank) {
                    idx_t id = xids ? xids[i] : ntotal + i;
                    size_t offset = invlists->add_entry(list_no, id, (const uint8_t*)xi);
                    dm_adder.add(i * num_centroids + j, list_no, offset);
                    n_add++;
                } else if (rank == 0 && list_no == -1) {
                    dm_adder.add(i * num_centroids + j, -1, 0);
                }
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}


void IndexIVFDiskANN::build_disk_ann(int n_list){
    // Validate parameters

    if (data_type.empty() || dist_fn.empty() || data_path.empty() || index_path_prefix.empty()) {
        throw std::invalid_argument("Missing required parameters for building DiskANN index.");
    }

    assert(codebook_prefix == "");
    
    const std::string in_list_data_path = data_path + "_"+ std::to_string(n_list) + ".bin";
    const std::string in_list_index_path_prefix = index_path_prefix + "_"+ std::to_string(n_list) + ".bin";
    
    std::cout << "build data_path:" << data_path << std::endl;
    std::cout << "build index_path_prefix:" << index_path_prefix << std::endl;
    std::cout << "build in_list_data_path:" << in_list_data_path << std::endl;
    std::cout << "build in_list_index_path_prefix:" << in_list_index_path_prefix << std::endl;

    try {
        bool use_filters = !label_file.empty();
        diskann::Metric metric;

        // Determine the metric type based on dist_fn
        if (dist_fn == "l2") {
            metric = diskann::Metric::L2;
        } else if (dist_fn == "mips") {
            metric = diskann::Metric::INNER_PRODUCT;
        } else if (dist_fn == "cosine") {
            metric = diskann::Metric::COSINE;
        } else {
            throw std::invalid_argument("Error: Only l2, mips, and cosine distance functions are supported.");
        }

        // Validation for append_reorder_data
        if (append_reorder_data) {
            if (disk_PQ == 0) {
                throw std::invalid_argument("Error: Appending reorder data requires disk_PQ to be non-zero.");
            }
            if (data_type != "float") {
                throw std::invalid_argument("Error: Appending reorder data is only supported for float data type.");
            }
        }

        // Prepare parameters string
        std::string params = std::to_string(R) + " " +
                             std::to_string(L) + " " +
                             std::to_string(B) + " " +
                             std::to_string(M) + " " +
                             std::to_string(num_threads) + " " +
                             std::to_string(disk_PQ) + " " +
                             std::to_string(append_reorder_data) + " " +
                             std::to_string(build_PQ) + " " +
                             std::to_string(QD);

        // Call the appropriate diskann::build_disk_index based on data_type
        if (data_type == "int8") {
            diskann::build_disk_index<int8_t>(
                in_list_data_path.c_str(), in_list_index_path_prefix.c_str(), params.c_str(), metric,
                use_opq, codebook_prefix, use_filters, label_file,
                universal_label, filter_threshold, Lf);
        } else if (data_type == "uint8") {
            diskann::build_disk_index<uint8_t>(
                in_list_data_path.c_str(), in_list_index_path_prefix.c_str(), params.c_str(), metric,
                use_opq, codebook_prefix, use_filters, label_file,
                universal_label, filter_threshold, Lf);
        } else if (data_type == "float") {
            diskann::build_disk_index<float>(
                in_list_data_path.c_str(), in_list_index_path_prefix.c_str(), params.c_str(), metric,
                use_opq, codebook_prefix, use_filters, label_file,
                universal_label, filter_threshold, Lf);
        } else {
            throw std::invalid_argument("Error: Unsupported data type");
        }

        std::cout << "DiskANN index built successfully." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Index build failed: " << e.what() << std::endl;
        throw;
    }
}


void IndexIVFDiskANN::merge_ivf_diskann(){
    const size_t page_size = 4096; // Page size
    size_t current_offset = 0;

    std::string merged_file_path = index_path_prefix + "_merged.index";
    std::ofstream merged_file(merged_file_path, std::ios::binary);

    if (!merged_file.is_open()) {
        throw std::runtime_error("Failed to open merged index file: " + merged_file_path);
    }

    for (int i = 0; i < nlist; ++i) {
        // Align to page boundary
        size_t aligned_offset = (current_offset + page_size - 1) & ~(page_size - 1);
        if (aligned_offset > current_offset) {
            std::vector<char> padding(aligned_offset - current_offset, 0);
            merged_file.write(padding.data(), padding.size());
        }

        std::cout << "aligned_offset: " << aligned_offset << std::endl;
        if(aligned_offset % page_size != 0){
            std::cout << "Not aligned!!!!!!!!!!!!!!!!!!!!" << std::endl;
        }



        // Write the per-list index chunk
        std::string list_file_path = index_path_prefix + "_" + std::to_string(i) + ".bin" + "_disk.index";
        std::ifstream list_file(list_file_path, std::ios::binary | std::ios::ate);

        if (!list_file.is_open()) {
            throw std::runtime_error("Failed to open list file: " + list_file_path);
        }

        std::streamsize file_size = list_file.tellg();
        if (file_size < 0) {
            throw std::runtime_error("Failed to get file size for: " + list_file_path);
        }

        current_offset = aligned_offset;
        disk_index_offset[i] = current_offset;

        list_file.seekg(0, std::ios::beg);

        std::vector<char> buffer(file_size);
        list_file.read(buffer.data(), file_size);
        merged_file.write(buffer.data(), buffer.size());

        current_offset += file_size;

        list_file.close();
    }

    merged_file.close();

    std::cout << "Merge completed. Merged index saved to: " << merged_file_path << std::endl;
}



namespace{

    template <typename T>
    void save_bin_data(int n_list, faiss::InvertedLists* invertedLists, const std::string& data_path_nlist) {
        size_t list_size = invertedLists->list_size(n_list);
        if (list_size == 0) {
            std::cerr << "List " << n_list << " is empty. Skipping." << std::endl;
            return;
        }
        size_t code_size = invertedLists->code_size;

        // Ensure the payload represents float values
        if (code_size % sizeof(float) != 0) {
            throw std::runtime_error("Code size is not a multiple of sizeof(float). Invalid data format.");
        }
        
        size_t num_floats_per_code = code_size / sizeof(float);
        size_t total_floats = list_size * num_floats_per_code;

        std::cout << "num_floats_per_code:" << num_floats_per_code << std::endl;
        std::cout << "list_size:" << list_size << std::endl;
        
        std::vector<float> float_buffer(total_floats);

        const uint8_t* codes = invertedLists->get_codes(n_list);
        std::memcpy(float_buffer.data(), codes, total_floats * sizeof(float));

        // Convert to target type T
        std::vector<T> buffer(total_floats);
        std::transform(float_buffer.begin(), float_buffer.end(), buffer.begin(), [](float val) {
            return static_cast<T>(val);
        });

        std::cout << "save file path:" << data_path_nlist << std::endl;

        // Open the output file
        std::ofstream writer(data_path_nlist, std::ios::binary);
        if (!writer.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_path_nlist);
        }

        // Write npts and ndims
        int32_t npts = static_cast<int32_t>(list_size);
        int32_t ndims = static_cast<int32_t>(num_floats_per_code);
        writer.write(reinterpret_cast<const char*>(&npts), sizeof(int32_t));
        writer.write(reinterpret_cast<const char*>(&ndims), sizeof(int32_t));

        // Write buffer contents
        writer.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(T));
        writer.close();
    }

    void save_invlist_bindata(int n_list, faiss::InvertedLists* invertedLists, const std::string& data_type, const std::string& data_path) {
        std::string data_path_nlist = data_path + "_" + std::to_string(n_list) + ".bin";

        if (data_type == "int8") {
            save_bin_data<int8_t>(n_list, invertedLists, data_path_nlist);
        } else if (data_type == "uint8") {
            save_bin_data<uint8_t>(n_list, invertedLists, data_path_nlist);
        } else if (data_type == "float") {
            save_bin_data<float>(n_list, invertedLists, data_path_nlist);
        } else {
            throw std::invalid_argument("Error: Unsupported data type");
        }
    }

}


// Add and train DiskAnn
void IndexIVFDiskANN::add_with_ids(idx_t n, const float* x, const idx_t* xids) {    
    idx_t k = this->assign_replicas;
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k]);
    quantizer->assign(n, x, coarse_idx.get(), k);
    add_core(n, x, xids, coarse_idx.get());

    if(this->add_batch_num-- == 0){
        //1. save each list to disk
        for(int i = 0; i < nlist; i++){
            save_invlist_bindata(i, this->invlists, data_type, data_path);
        }

        //2. call diskann function
        for(int i = 0; i < nlist; i++){
            build_disk_ann(i);
        }

        //3. merge
        merge_ivf_diskann();

        for(int i = 0; i< nlist; i++){
            std::cout << "disk_offset:" << 
                        "origin: " << this->disk_index_offset[i] <<
                        "  in MB:" << (this->disk_index_offset[i]/(1024*1024)) << std::endl;
        }
    }
}


void IndexIVFDiskANN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

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

    
    // handle parallelization at level below (or don't run in parallel at
    // all)
    sub_search_func(n, x, distances, labels, &indexIVF_stats);
    
}


void IndexIVFDiskANN::search_preassigned(
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
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;

    if(this->data_type == "float"){
        search_graph<float>(n, k, keys, coarse_dis,distances,labels);
    }else if(this->data_type == "uint8"){
        search_graph<uint8_t>(n, k, keys, coarse_dis,distances,labels);
    }else if(this->data_type == "int8"){
        search_graph<int8_t>(n, k, keys, coarse_dis,distances,labels);
    }
    




}


template<typename T>
void IndexIVFDiskANN::load_diskann(){

    // diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    // if (beamwidth <= 0)
    //     diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    // else
    //     diskann::cout << " beamwidth: " << beamwidth << std::flush;
    // if (search_io_limit == std::numeric_limits<uint32_t>::max())
    //     diskann::cout << "." << std::endl;
    // else
    //     diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    // std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // // load query bin
    // T *query = nullptr;
    // uint32_t *gt_ids = nullptr;
    // float *gt_dists = nullptr;
    // size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    // diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // bool filtered_search = false;
    // if (!query_filters.empty())
    // {
    //     filtered_search = true;
    //     if (query_filters.size() != 1 && query_filters.size() != query_num)
    //     {
    //         std::cout << "Error. Mismatch in number of queries and size of query "
    //                      "filters file"
    //                   << std::endl;
    //         return -1; // To return -1 or some other error handling?
    //     }
    // }

    // bool calc_recall_flag = false;
    // if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    // {
    //     diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    //     if (gt_num != query_num)
    //     {
    //         diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    //     }
    //     calc_recall_flag = true;
    // }
}

// No need to load vectors anymore
template<typename T,  typename LabelT>
void IndexIVFDiskANN::search_graph(
        idx_t n,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels) const{
    
    diskann::Metric metric = diskann::Metric::L2;
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

    std::string query_file = "/mnt/d/VectorDB/sift/sift/sift_query.fbin";
    // Skip ground truth for now
    std::cout << "query_file: " << query_file << std::endl;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);


    std::shared_ptr<AlignedFileReader> reader = nullptr;
    reader.reset(new LinuxAlignedFileReader());

    std::vector<std::unique_ptr<diskann::PQFlashIndex<T, LabelT>>> _pFlashIndices;
    _pFlashIndices.reserve(nlist); // Reserve to avoid repeated allocations

    for (size_t i = 0; i < nlist; ++i) {
        _pFlashIndices.emplace_back(
            std::make_unique<diskann::PQFlashIndex<T, LabelT>>(reader, metric)
        );
    }

    for(int i = 0; i < nlist; i++)
    {      
        size_t offset = this->disk_index_offset[i];

        std::cout << "\n\n\n offset:" << offset ;
        std::cout << " nlist:" << nlist << std::endl;
        std::cout << "index_path_prefix:" << index_path_prefix << "  i:" << i << std::endl;

        std::string index_list_path_prefix = index_path_prefix  + "_" + std::to_string(i) + ".bin";

        std::cout << "index_list_path_prefix:" << index_list_path_prefix << "\n\n\n\n\n"<< std::endl;
    
        int res = _pFlashIndices[i]->load(1, index_list_path_prefix.c_str(), (index_path_prefix+"_merged.index").c_str() , offset);
        //int res = 0;
        if (res != 0)
        {
            std::cout << "Line 509 Error res:" << res << std::endl; 
            return;
        }
    }   



    
    std::vector<uint32_t> Lvec;
    // at recall 100

    Lvec.push_back(50);
    Lvec.push_back(20);
    Lvec.push_back(20);
    Lvec.push_back(20);
    Lvec.push_back(10);

    const uint32_t recall_at = 50 + 20 + 20 + 20 + 10;

//     std::vector<uint32_t> node_list;
//     diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
//     _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
//     // if (num_nodes_to_cache > 0)
//     //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
//     //     num_threads, node_list);
//     _pFlashIndex->load_cache_list(node_list);
//     node_list.clear();
//     node_list.shrink_to_fit();

//     omp_set_num_threads(num_threads);

//     uint64_t warmup_L = 20;
//     uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
//     T *warmup = nullptr;

//      if (WARMUP)
//     {
//         if (file_exists(warmup_query_file))
//         {
//             diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
//         }
//         else
//         {
//             warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
//             warmup_dim = query_dim;
//             warmup_aligned_dim = query_aligned_dim;
//             diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
//             std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
//             std::random_device rd;
//             std::mt19937 gen(rd());
//             std::uniform_int_distribution<> dis(-128, 127);
//             for (uint32_t i = 0; i < warmup_num; i++)
//             {
//                 for (uint32_t d = 0; d < warmup_dim; d++)
//                 {
//                     warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
//                 }
//             }
//         }
//         diskann::cout << "Warming up index... " << std::flush;
//         std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
//         std::vector<float> warmup_result_dists(warmup_num, 0);

// #pragma omp parallel for schedule(dynamic, 1)
//         for (int64_t i = 0; i < (int64_t)warmup_num; i++)
//         {
//             _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
//                                              warmup_result_ids_64.data() + (i * 1),
//                                              warmup_result_dists.data() + (i * 1), 4);
//         }
//         diskann::cout << "..done" << std::endl;
//     }

    /*
    Coarse search
    */

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    // std::string recall_string = "Recall@" + std::to_string(recall_at);
    // diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
    //               << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
    //               << "CPU (s)";
    // if (calc_recall_flag)
    // {
    //     diskann::cout << std::setw(16) << recall_string << std::endl;
    // }
    // else
    //     diskann::cout << std::endl;

    // Each query needs four slots for results

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    
    for(int i = 0; i < Lvec.size(); i++){
        // query_result_ids[i].resize(Lvec[i]*query_num);
        // query_result_dists[i].resize(Lvec[i]*query_num);
        query_result_ids[i].resize(recall_at*query_num);
        query_result_dists[i].resize(recall_at*query_num);
    }
    
    uint32_t optimized_beamwidth = 2;
    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    auto stats = new diskann::QueryStats[query_num*Lvec.size()];

    double best_recall = 0.0;
    for (int64_t i = 0; i < (int64_t)query_num; i++){

        uint32_t cummul = 0;
        for(int64_t j = 0; j < Lvec.size(); j++){

            uint32_t qno = i*Lvec.size() + j;
            std::cout << "qno:" << qno<< " key:" << keys[qno] << std::endl;
            _pFlashIndices[keys[qno]]->cached_beam_search(query + (i * query_aligned_dim), Lvec[j], Lvec[j],
                                            query_result_ids_64.data() + (i * recall_at) + cummul ,
                                            query_result_dists[j].data() + (i * Lvec.size())  + cummul,
                                            optimized_beamwidth, false, stats + qno);
            cummul += Lvec[j];

            for(int disno = 0; disno < Lvec[j]; disno++){
                std::cout << "dis: " << (query_result_ids_64.data() + (i * recall_at))[disno] << "\n";
                std::cout << "ids: " << (query_result_dists[j].data() + (i * Lvec.size()))[disno] << "\n";
            }
        }
    }


//     for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
//     {
//         uint32_t L = Lvec[test_id];

//         optimized_beamwidth = beamwidth;

//         query_result_ids[test_id].resize(recall_at * query_num);
//         query_result_dists[test_id].resize(recall_at * query_num);

//         auto stats = new diskann::QueryStats[query_num];

//         std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
//         auto s = std::chrono::high_resolution_clock::now();

//         idx_t* coarse_ids;

//         for (int64_t i = 0; i < (int64_t)query_num; i++){

//             uint32_t cummul = 0;
//             for(int64_t j = 0; j < Lvec.size(); j++){
//                 _pFlashIndex[]->cached_beam_search(query + (i * query_aligned_dim), Lvec[j], L,
//                                                 query_result_ids_64.data() + (i * recall_at) + cummul ,
//                                                 query_result_dists[test_id].data() + (i * recall_at)  + cummul,
//                                                 optimized_beamwidth, use_reorder_data, stats + i);
//                 cummul += Lvec[j];
//             }
//         }

// // #pragma omp parallel for schedule(dynamic, 1)
// //         for (int64_t i = 0; i < (int64_t)query_num; i++)
// //         {
// //             if (!filtered_search)
// //             {
// //                 _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
// //                                                  query_result_ids_64.data() + (i * recall_at),
// //                                                  query_result_dists[test_id].data() + (i * recall_at),
// //                                                  optimized_beamwidth, use_reorder_data, stats + i);
// //             }
// //         }


//         auto e = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> diff = e - s;
//         double qps = (1.0 * query_num) / (1.0 * diff.count());

//         diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
//                                                    query_num, recall_at);

//         auto mean_latency = diskann::get_mean_stats<float>(
//             stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

//         auto latency_999 = diskann::get_percentile_stats<float>(
//             stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

//         auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
//                                                           [](const diskann::QueryStats &stats) { return stats.n_ios; });

//         auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
//                                                          [](const diskann::QueryStats &stats) { return stats.cpu_us; });

//         double recall = 0;
//         if (calc_recall_flag)
//         {
//             recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
//                                                query_result_ids[test_id].data(), recall_at, recall_at);
//             best_recall = std::max(recall, best_recall);
//         }

//         diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
//                       << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
//                       << std::setw(16) << mean_cpuus;
//         if (calc_recall_flag)
//         {
//             diskann::cout << std::setw(16) << recall << std::endl;
//         }
//         else
//             diskann::cout << std::endl;
//         delete[] stats;
//     }
}



} // namespace faiss
