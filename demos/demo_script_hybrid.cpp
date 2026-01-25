#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexDiskV.h>

#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <random>
#include <omp.h>

using idx_t = faiss::idx_t;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef std::map<std::string, std::string> ConfigMap;
void read_config(const std::string& path, ConfigMap& config) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open config file: " << path << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(in, line)) {
        auto sharp = line.find('#');
        if (sharp != std::string::npos) line = line.substr(0, sharp);
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty()) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        config[key] = val;
    }
}


std::vector<size_t> parse_size_t_list(const std::string& s) {
    std::vector<size_t> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) res.push_back(std::stoul(item));
    return res;
}
std::vector<float> parse_float_list(const std::string& s) {
    std::vector<float> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) res.push_back(std::stof(item));
    return res;
}

float* load_and_convert_to_float(const char* fname, size_t* d_out, size_t* n_out, size_t batch_size = 1000000) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        perror("");
        abort();
    }

    // Read vector dimension
    int d;
    size_t unuse_result = fread(&d, sizeof(int), 1, f);
    assert((d > 0 && d < 1000000) || !"Unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    // Verify file size matches expectation
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;

    // Each vector consists of 1 int (dimension header) + d uint8 values
    size_t per_vector_size = sizeof(int) + d * sizeof(uint8_t);
    assert(sz % per_vector_size == 0 || !"Weird file size");

    size_t n = sz / per_vector_size; // Number of vectors
    *d_out = d;
    *n_out = n;

    std::cout << "d: " << d << "  n:" << n << std::endl;
    // Allocate final float storage
    float* result = new float[n * d];

    size_t vectors_left = n;
    size_t offset = 0; // Track current write position in result

    while (vectors_left > 0) {
        size_t current_batch_size = std::min(batch_size, vectors_left);

        // Temporary buffer: 1 int header followed by d uint8 values
        std::vector<uint8_t> temp(per_vector_size * current_batch_size);
        size_t nr = fread(temp.data(), sizeof(uint8_t), temp.size(), f);
        std::cout << "vector size:"<<  current_batch_size <<" nr:" << nr << "  temp size:" << temp.size() << std::endl;
        assert(nr == temp.size() || !"Could not read batch");

        // Convert and write into result
        for (size_t i = 0; i < current_batch_size; i++) {
            // Skip the dimension header (1 int)
            const uint8_t* data_ptr = temp.data() + i * per_vector_size + sizeof(int);

            // Convert uint8 to float
            for (size_t j = 0; j < d; j++) {
                result[offset + i * d + j] = static_cast<float>(data_ptr[j]);
            }
        }

        offset += current_batch_size * d; // Advance write offset
        vectors_left -= current_batch_size;
    }

    fclose(f);
    return result;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    size_t unuse_result = fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    uint32_t n, d;
    size_t unuse_result = fread(&n, sizeof(uint32_t), 1, f);
    unuse_result = fread(&d, sizeof(uint32_t), 1, f);

    //std::cout << "n: " << n << " d:" << d << "\n";

    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    assert((n > 0 && n < 1000000000) || !"unreasonable number of vectors");

    size_t total_size = (size_t)n * d;  // Guard against overflow
    float* x = new float[total_size];

    size_t chunk_size = 1024;
    size_t chunk_elements = chunk_size * d;  // Elements per read chunk
    float* buffer = new float[chunk_elements];

    size_t elements_read = 0;
    while (elements_read < total_size) {
        size_t elements_to_read = std::min(chunk_elements, total_size - elements_read);
        size_t nr = fread(buffer, sizeof(float), elements_to_read, f);
        assert(nr == elements_to_read || !"could not read whole file");

        // Copy data into the final array
        std::copy(buffer, buffer + nr, x + elements_read);
        elements_read += nr;
    }

    fclose(f);

    *d_out = d;
    *n_out = n;
    return x;
}

int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fbin_read(fname, d_out, n_out);
}

float* int8bin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    uint32_t n, d;
    size_t unuse_result = fread(&n, sizeof(uint32_t), 1, f);
    unuse_result = fread(&d, sizeof(uint32_t), 1, f);

    //std::cout << "n: " << n << " d:" << d << "\n";

    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    assert((n > 0 && n < 1000000000) || !"unreasonable number of vectors");

    size_t total_size = (size_t)n * d;
    float* x = new float[total_size];  // Return buffer as float array

    size_t chunk_size = 1024;
    size_t chunk_elements = chunk_size * d;
    int8_t* buffer = new int8_t[chunk_elements];  // Temporary int8 buffer

    size_t elements_read = 0;
    while (elements_read < total_size) {
        size_t elements_to_read = std::min(chunk_elements, total_size - elements_read);
        size_t nr = fread(buffer, sizeof(int8_t), elements_to_read, f);
        assert(nr == elements_to_read || !"could not read whole file");

        // Convert int8 to float and store in result
        for (size_t i = 0; i < nr; ++i) {
            x[elements_read + i] = static_cast<float>(buffer[i]);
        }
        elements_read += nr;
    }

    delete[] buffer;
    fclose(f);

    *d_out = d;
    *n_out = n;
    return x;
}


float* uint8bin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    uint32_t n, d;
    size_t unuse_result = fread(&n, sizeof(uint32_t), 1, f);
    unuse_result = fread(&d, sizeof(uint32_t), 1, f);

    //std::cout << "n: " << n << " d:" << d << "\n";

    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    assert((n > 0 && n < 1000000000) || !"unreasonable number of vectors");

    size_t total_size = (size_t)n * d;
    float* x = new float[total_size];  // Return buffer as float array

    size_t chunk_size = 1024;
    size_t chunk_elements = chunk_size * d;
    uint8_t* buffer = new uint8_t[chunk_elements];  // Temporary uint8 buffer

    size_t elements_read = 0;
    while (elements_read < total_size) {
        size_t elements_to_read = std::min(chunk_elements, total_size - elements_read);
        size_t nr = fread(buffer, sizeof(uint8_t), elements_to_read, f);
        assert(nr == elements_to_read || !"could not read whole file");

        // Convert uint8 values to float
        for (size_t i = 0; i < nr; ++i) {
            x[elements_read + i] = static_cast<float>(buffer[i]);
        }
        elements_read += nr;
    }

    delete[] buffer;
    fclose(f);

    *d_out = d;
    *n_out = n;
    return x;
}



float* load_dataset_vectors(const std::string& path, const std::string& fmt, size_t* d, size_t* n) {
    if (fmt == "fvecs") return fvecs_read(path.c_str(), d, n);
    if (fmt == "ivecs") return load_and_convert_to_float(path.c_str(), d, n);
    if (fmt == "bvecs") return load_and_convert_to_float(path.c_str(), d, n);
    if (fmt == "fbin")  return fbin_read(path.c_str(), d, n);
    if (fmt == "i8bin")  return int8bin_read(path.c_str(), d, n);
    if (fmt == "u8bin")  return uint8bin_read(path.c_str(), d, n);
    std::cerr << "Dataset only support fvecs|ivecs|bvecs|fbin|i8bin|u8bin, not support " << fmt << std::endl;
    exit(1);
}

int* load_groundtruth(const std::string& path, const std::string& fmt, size_t* d, size_t* n){
    if (fmt == "ivecs") return ivecs_read(path.c_str(), d, n);
    if (fmt == "ibin")  return ibin_read(path.c_str(), d, n);
    std::cerr << "ground truth only support ivecs or ibin, not support" << fmt << std::endl;
    exit(1);
}

void print_index_info(size_t k, size_t nq, size_t nprobe, bool reset = true){
    (void)nprobe;

    const double inv_nq = nq ? 1.0 / static_cast<double>(nq) : 0.0;

    const size_t sl  = faiss::indexDiskV_stats.searched_lists;
    const size_t svf = faiss::indexDiskV_stats.searched_vector_full;
    const size_t svp = faiss::indexDiskV_stats.searched_vector_partial;
    const size_t spf = faiss::indexDiskV_stats.searched_page_full;
    const size_t spp = faiss::indexDiskV_stats.searched_page_partial;
    const size_t rf  = faiss::indexDiskV_stats.requests_full;
    const size_t rp  = faiss::indexDiskV_stats.requests_partial;
    const size_t cla = faiss::indexDiskV_stats.cached_list_access;
    const size_t cva = faiss::indexDiskV_stats.cached_vector_access;

    const double me1 = faiss::indexDiskV_stats.memory_1_elapsed.count() / 1000.0;
    const double me2 = faiss::indexDiskV_stats.memory_2_elapsed.count() / 1000.0;
    const double me3 = faiss::indexDiskV_stats.memory_3_elapsed.count() / 1000.0;
    const double memory_time_cost = (me1 + me2 + me3) * inv_nq;

    const double dfe = faiss::indexDiskV_stats.disk_full_elapsed.count() / 1000.0;
    const double dpe = faiss::indexDiskV_stats.disk_partial_elapsed.count() / 1000.0;
    const double ce  = faiss::indexDiskV_stats.coarse_elapsed.count() / 1000.0;
    const double re  = faiss::indexDiskV_stats.rank_elapsed.count() / 1000.0;
    const double pe  = faiss::indexDiskV_stats.pq_elapsed.count() / 1000.0;

    std::cout << "K=" << k << std::endl;
    std::cout << "Scanned lists per query :" << static_cast<double>(sl) * inv_nq << std::endl;
    std::cout << "Memory_time_cost        :" << memory_time_cost << std::endl;
    std::cout << "disk_full_elapsed       :" << dfe * inv_nq << std::endl;
    std::cout << "disk_partial_elapsed    :" << dpe * inv_nq << std::endl;
    std::cout << "coarse_elapsed          :" << ce * inv_nq << std::endl;
    std::cout << "rank_elapsed            :" << re * inv_nq << std::endl;
    std::cout << "pq_elapsed              :" << pe * inv_nq << std::endl;
    std::cout << "searched_vector_full    :" << static_cast<double>(svf) * inv_nq << std::endl;
    std::cout << "searched_vector_partial :" << static_cast<double>(svp) * inv_nq << std::endl;
    std::cout << "searched_page_full      :" << static_cast<double>(spf) * inv_nq << std::endl;
    std::cout << "searched_page_partial   :" << static_cast<double>(spp) * inv_nq << std::endl;
    std::cout << "requests_full           :" << static_cast<double>(rf) * inv_nq << std::endl;
    std::cout << "requests_partial        :" << static_cast<double>(rp) * inv_nq << std::endl;
    std::cout << "cached lists            :" << static_cast<double>(cla) * inv_nq << std::endl;
    std::cout << "cached vectors          :" << static_cast<double>(cva) * inv_nq << std::endl;

    if(reset)
        faiss::indexDiskV_stats.reset();
}





void distribute_vectors(float* xb, size_t nb, size_t d, 
                        std::vector<std::vector<float>>& xbs, 
                        std::vector<std::vector<idx_t>>& xids,
                        size_t partitions) {
    xbs.resize(partitions);
    xids.resize(partitions);
    
    // Preallocate capacity assuming uniform distribution
    for (size_t i = 0; i < partitions; i++) {
        xbs[i].reserve((nb / partitions + 1) * d);  // Reserve enough space
        xids[i].reserve(nb/partitions + 1);
    }

    // Iterate over all vectors

    for (size_t i = 0; i < nb; i++) {
        size_t target_partition = i % partitions;
        size_t old_size = xbs[target_partition].size();
        xbs[target_partition].resize(old_size + d);  // Expand storage directly
        xids[target_partition].push_back(i);
        std::memcpy(&xbs[target_partition][old_size], xb + i * d, d * sizeof(float));  // Copy data directly
    }
}


void sep_build(float* xb_i, idx_t* xids, size_t nb_i, size_t d, int ratio, size_t partition_num, size_t order,
    const std::string& index_store_path,    
    const std::string& disk_store_path,
    const std::string& centroid_index_path,
    int nlist, int m, int nbits, int replicas, float shrink_replicas,
    float estimate_factor, float prune_factor, std::string metric_type, std::string vector_type, int top_clusters,
    bool verbose, int memory_graph_ef_construction, int memory_graph_ef_search, int memory_graph_M)
{
    double t0 = elapsed();
    std::vector<float> trainvecs(nb_i / ratio * d);
    int base_number = 0;
    srand(static_cast<int>(time(0)));
    for (int i = 0; i < nb_i / ratio; i++) {
        int rng = (rand() % (ratio));
        for (int j = 0; j < d; j++) {
            trainvecs[d * i + j] = xb_i[((size_t)(rng + base_number) * (size_t)d + j)];
        }
        base_number += ratio;
    }
    // TODO 
    faiss::MetricType mt = metric_type == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

    if(vector_type != "uint8" && vector_type != "float" && vector_type != "int"){
        FAISS_THROW_MSG("Only support uint8, float, int");
    }

    faiss::IndexFlat quantizer(d, mt);
    faiss::IndexDiskV index(&quantizer, d, nlist, m, nbits, top_clusters, 
        estimate_factor, prune_factor, index_store_path, vector_type, mt);
    faiss::IndexHNSWFlat cluster_trainer(d, 16, mt);
    cluster_trainer.hnsw.efConstruction = 75;
    cluster_trainer.hnsw.efSearch = 150;
    index.clustering_index = &cluster_trainer;
    index.set_assign_replicas(replicas);
    index.set_centroid_index_path(centroid_index_path);
    index.set_select_lists_mode();
    index.set_in_list_clustering();
    index.set_soaring();
    index.set_shrink_replica(shrink_replicas);
    index.set_multi_index(order, partition_num);
    index.set_memory_graph_build_params(
        memory_graph_ef_construction, memory_graph_ef_search, memory_graph_M);
    index.verbose = verbose;
    printf("     [%.3f s] train\n", elapsed() - t0);
    index.train(nb_i / ratio, trainvecs.data());
    printf("     [%.3f s] train finished\n", elapsed() - t0);
    printf("     [%.3f s] add\n", elapsed() - t0);
    int nadd = 1;                       // TODO: make configurable
    index.add_batch_num = nadd;
    for(int i = 0; i < nadd; i++){
        index.add_with_ids(nb_i/nadd, xb_i + (nb_i/nadd * i) * d, nullptr);
    }
    printf("     [%.3f s] add finished\n", elapsed() - t0);
    printf("     [%.3f s] start to reorg\n", elapsed() - t0);
    printf("     [%.3f s] reorg finished\n", elapsed() - t0);
    faiss::write_index(&index, disk_store_path.c_str());
    printf("     [%.3f s] written to disk\n", elapsed() - t0);
}





void disk_build(
    const char* base_filepath, int nb, int d, int ratio, 
    const std::string& index_store_path,    
    const std::string& disk_store_path,
    const std::string& centroid_index_path,
    int partitions,
    int nlist, int m, int nbits, int replicas, float shrink_replicas,
    float estimate_factor, float prune_factor, std::string metric_type, std::string dataset_fmt,  std::string vector_type,
    int build_threads, int top_clusters = 3, bool verbose = false,
    int memory_graph_ef_construction = 40, int memory_graph_ef_search = 16,
    int memory_graph_M = 16
) {
    double t0 = elapsed();
    size_t dd, nt;
    //float* xb = load_and_convert_to_float(base_filepath, &dd, &nt);

    float* xb = load_dataset_vectors(base_filepath, dataset_fmt, &dd, &nt);

    std::vector<std::vector<float>> xbs;
    std::vector<std::vector<idx_t>> xids;
    distribute_vectors(xb, nb, d, xbs, xids, partitions);
    delete[] xb;
    omp_set_num_threads(build_threads);
    for(int i = 0; i < partitions; i++) {
        std::string index_store_path_i = disk_store_path + "_" + std::to_string(i);
        std::string disk_store_path_i = disk_store_path + "_" + std::to_string(i) + ".index";
        std::string centroid_index_path_i = centroid_index_path + "_" + std::to_string(i);
        double t1 = elapsed();
        sep_build(xbs[i].data(), xids[i].data(), xbs[i].size()/d, d, ratio, partitions, i,
            index_store_path_i, disk_store_path_i, centroid_index_path_i,
            nlist, m, nbits, replicas, shrink_replicas, estimate_factor, prune_factor, metric_type, vector_type, top_clusters,
            verbose, memory_graph_ef_construction, memory_graph_ef_search, memory_graph_M);
        std::cout << "Thread " << omp_get_thread_num() 
                      << " finished processing index " << i 
                      << " in " << (elapsed() - t1) << " seconds." << std::endl;
                      printf("[%.3f s] batch %d written to disk\n", elapsed() - t0, i);
    }
}



void load_indices(faiss::IndexDiskV*& index, 
                  const std::string& disk_store_path,
                  const std::string& centroid_index_path,     
                  float estimate_factor, 
                  float estimate_factor_partial,
                  float estimate_factor_high_dim,
                  float prune_factor,
                  size_t top,
                  int replicas,
                  int max_continous_pages,
                  int full_decode_volume,
                  int partial_one_decode_volume,
                  int partial_two_decode_volume,
                  int submit_per_round,
                  bool verbose
) {
    std::string index_meta_data_path =  disk_store_path + ".index";
    std::string index_vector_path = disk_store_path + ".clustered";
    index = dynamic_cast<faiss::IndexDiskV*>(faiss::read_index(index_meta_data_path.c_str()));
    index->select_lists_path = index_vector_path;
    index->disk_path = index_vector_path;
    index->set_centroid_index_path(centroid_index_path);
    index->load_hnsw_centroid_index();
    index->set_top(top);
    index->set_assign_replicas(replicas);
    index->set_estimate_factor(estimate_factor);
    index->set_estimate_factor_partial(estimate_factor_partial);
    index->set_estimate_factor_high_dim(estimate_factor_high_dim);
    index->set_prune_factor(prune_factor);
    index->set_max_continous_pages(max_continous_pages);
    index->set_ahead_pq_volumes(full_decode_volume, partial_one_decode_volume, partial_two_decode_volume);
    index->set_submit_per_round(submit_per_round);
    if (verbose) {
        std::cout << "top:" << top << " \nreplicas:" << replicas 
                  << " \nestimate_factor:" << estimate_factor 
                  << " \nestimate_factor_partial:" << estimate_factor_partial 
                  << " \nestimate_factor_high_dim:" << estimate_factor_high_dim 
                  << " \nprune_factor:" << prune_factor 
                  << " \nmax_continous_pages:" << max_continous_pages
                  << " \nfull_decode_volume:" << full_decode_volume
                  << " \npartial_one_decode_volume:" << partial_one_decode_volume
                  << " \npartial_two_decode_volume:" << partial_two_decode_volume
                  << " \nsubmit_per_round:" << submit_per_round
                  << std::endl;
    }

}


void disk_search(const char* query_filepath, const char* ground_truth_filepath, 
            int nq, int d, int k, int k_per_partition,
            const std::string& disk_store_path,
            const std::string& centroid_index_path,
            int partitions,
            float estimate_factor, 
            float estimate_factor_high_dim,
            float prune_factor, 
            size_t top, 
            int replicas, 
            
            std::vector<size_t> nprobes,
            std::vector<float> estimate_factors_partial,

            int max_continous_pages,
            int full_decode_volume,
            int partial_one_decode_volume,
            int partial_two_decode_volume,
            int submit_per_round,
            const std::string& queryset_fmt, 
            const std::string& truthset_fmt,
            int search_threads,
            int cache_vectors,
            int query_for_warm_up,
            bool verbose,
            // ---- NEW: split loading across two disks ----
            int disk1_count,
            const std::string& disk1_store_base,
            const std::string& disk1_centroid_base,
            const std::string& disk2_store_base,
            const std::string& disk2_centroid_base,
            // ---- NEW: per-disk search params for loading ----
            float est1, float est_hd1, float prune1, size_t top1, int replicas1,
            int maxp1, int fdv1, int pdv1, int ptv1, int spr1,
            float est2, float est_hd2, float prune2, size_t top2, int replicas2,
            int maxp2, int fdv2, int pdv2, int ptv2, int spr2,
            // ---- NEW: optional per-disk overrides for runtime loop ----
            int   nprobe1_opt, int nprobe2_opt,          // -1 means keep the loop's nprobe
            float partial1_opt, float partial2_opt       // <0 means keep the loop's tem_partial
            
) {
    double t0 = elapsed();
    size_t dd2; // dimension
    size_t nt2; // number of queries
    size_t nq2; // number of queries
    size_t kk;  // nearest neighbors in ground truth

    float* xq = load_dataset_vectors(query_filepath, queryset_fmt, &dd2, &nt2);
    int* gt_int = load_groundtruth(ground_truth_filepath, truthset_fmt, &kk, &nq2);

    int k_verify = k;

    if (verbose) {
        std::cout << "dimension:" << dd2 << "\nnumber of queries:" << nt2 
                  << "\nverifing queries:" << nq2 << "\nground truth:" << kk 
                  << "\nverifing neighbor:" << k_verify 
                  << "\nk_per_partition: " << k_per_partition << std::endl;
    }

    faiss::idx_t* gt = new faiss::idx_t[k_verify * nq];
    int jj=0;
    for (int i = 0; i < nq; i++) {
        for(int j = 0; j < k_verify; j++){
            gt[jj] =  static_cast<faiss::idx_t>(gt_int[i*kk+j]); //long int / int
            jj+=1;
        }
    }
    delete[] gt_int;

    std::vector<std::vector<idx_t>> Is(partitions);
    std::vector<std::vector<float>> Ds(partitions);
    for(int i = 0; i< partitions; i++){
        Is[i].resize(k_per_partition * nq);
        Ds[i].resize(k_per_partition * nq);
    }


    std::vector<faiss::IndexDiskV*> indices(partitions);
    for(int i = 0; i < partitions; i++){
        const bool on_disk1 = (i < disk1_count);
        const std::string& base_store = on_disk1 ? disk1_store_base : disk2_store_base;
        const std::string& base_cent  = on_disk1 ? disk1_centroid_base : disk2_centroid_base;

        std::string disk_store_path_i     = base_store + "_" + std::to_string(i);
        std::string centroid_index_path_i = base_cent  + "_" + std::to_string(i);

        // Use a dedicated parameter set per disk (consumed by load_indices)
        float   ef      = on_disk1 ? est1     : est2;
        float   ef_hd   = on_disk1 ? est_hd1  : est_hd2;
        float   pr      = on_disk1 ? prune1   : prune2;
        size_t  tp      = on_disk1 ? top1     : top2;
        int     reps    = on_disk1 ? replicas1: replicas2;
        int     maxp    = on_disk1 ? maxp1    : maxp2;
        int     fdv     = on_disk1 ? fdv1     : fdv2;
        int     pdv     = on_disk1 ? pdv1     : pdv2;
        int     ptv     = on_disk1 ? ptv1     : ptv2;
        int     spr     = on_disk1 ? spr1     : spr2;

        // Note: estimate_factors_partial[0] only seeds load defaults; runtime may override per disk
        float partial0 = estimate_factors_partial.empty() ? 1.0f : estimate_factors_partial[0];

        load_indices(
            indices[i], disk_store_path_i, centroid_index_path_i,
            ef, partial0, ef_hd,
            pr, tp, reps, maxp, fdv, pdv, ptv, spr, verbose
        );
    }
    omp_set_num_threads(search_threads);


    for(int i = 0; i < partitions; i++){
        printf("[%.3f s] Setting index %d I/O..\n", elapsed() - t0, i);
        indices[i]->warmUpListCache(1000, xq, 200, 0);
        //indices[i]->warmUpIndexMetadata(10000, xq, 200,indices[i]->nlist);
        indices[i]->warmUpAllIndexMetadata();
        indices[i]->initializeDiskIO(search_threads);

        size_t n_shard = 100000;
        indices[i]->set_cache_strategy(faiss::IMMEDIATELY_UPDATE);  // update when has read a vector

        size_t vector_warmed_up = indices[i]->warmUpVectorCacheShard(query_for_warm_up, xq, 100, 100, cache_vectors, 
                                                        estimate_factors_partial[0], search_threads ,n_shard);
        std::cout << "Warming up finished cached: " << vector_warmed_up << " vectors"<<std::endl;
        indices[i]->shutdownDiskIO(search_threads);
    }

    bool multi_filter = true;
    std::vector<double> search_times;
    std::vector<double> recalls;
    faiss::indexDiskV_stats.reset();

    std::cout << "Searching with " << nprobes.size() << " nprobes: ";
    for(int nprobe : nprobes){
        std::cout << nprobe << " ";
    }
    std::cout << std::endl;
    std::cout << "Searching with " << estimate_factors_partial.size() << " partial factors: ";
    for(float tem_partial : estimate_factors_partial){
        std::cout << tem_partial << " ";
    }
    std::cout << std::endl;

    for(int nprobe_loop : nprobes){  
        for(float partial_loop : estimate_factors_partial){
            for(int i = 0; i < partitions; i++){
                const bool on_disk1 = (i < disk1_count);
                int   applied_nprobe  = on_disk1
                                        ? (nprobe1_opt  > 0   ? nprobe1_opt  : nprobe_loop)
                                        : (nprobe2_opt  > 0   ? nprobe2_opt  : nprobe_loop);
                float applied_partial = on_disk1
                                        ? (partial1_opt >= 0.f ? partial1_opt : partial_loop)
                                        : (partial2_opt >= 0.f ? partial2_opt : partial_loop);

                indices[i]->nprobe = applied_nprobe;
                indices[i]->set_estimate_factor_partial(applied_partial);
            }

            double search_time = 0;
            for(int i = 0; i < partitions; i++){
                indices[i]->initializeDiskIO(search_threads);
                double t1 = elapsed();
                std::vector<float> sres(nq);

                if(multi_filter && i!=0){ 
                    for(int j = 0; j < nq; j++){
                        sres[j] = (Ds[i-1][(j+1)*k_per_partition-1]);
                    }
                    indices[i]->set_search_threshold(nq, sres);
                    indices[i]->q_indicator = new size_t[search_threads];
                }

                indices[i]->search(nq, xq, k_per_partition, Ds[i].data(), Is[i].data());

                double t2 = elapsed();
                search_time += t2 - t1;
                indices[i]->shutdownDiskIO(search_threads);
                if (verbose) {
                    print_index_info(k_verify, nq, (int)nprobe_loop, false);
                }
            }

            for(int i = 0; i < partitions; i++){
                indices[i]->clear_search_threshold();
                if(indices[i]->q_indicator!= nullptr){
                    delete[] indices[i]->q_indicator;
                    indices[i]->q_indicator = nullptr;
                }
            }

            int n2_100 = 0;
            if (verbose) {
                print_index_info(k_verify, nq, (int)nprobe_loop);
            }

            for (int i = 0; i < nq; i++) {
                std::unordered_set<idx_t> real_ids; 
                std::unordered_map<idx_t, int> umap;
                for (int j = 0; j < k_verify; j++) {
                    umap[gt[i * k_verify + j]] = 0;
                }
                for (int ii = 0; ii < partitions; ii++) {
                    for (int l = 0; l < k_per_partition; l++) {
                        idx_t real_id = Is[ii][i * k_per_partition + l];
                        real_ids.insert(real_id);
                    }
                }
                for (idx_t real_id : real_ids) {
                    if (umap.find(real_id) != umap.end()) {
                        n2_100++;
                    }
                }
            }
            printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k_verify));
            printf("QPS = %.4f\n", nq / search_time);

            double recall = n2_100 / float(nq * k_verify);
            search_times.push_back(search_time / nq * 1000);
            recalls.push_back(recall);
        }
    }

    std::cout << "Search times: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << search_times[i];
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "QPS         : [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << (1000.0 / search_times[i]);
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Recalls     : [";
    for (size_t i = 0; i < recalls.size(); i++) {
        std::cout << recalls[i];
        if (i < recalls.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Time Gap    : [";
    for(int i = 1; i < search_times.size(); i++){
        std::cout << search_times[i] - search_times[i-1];
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode: 0-build, 1-search> <config-file> [verbose]" << std::endl;
        return 1;
    }
    int type = atoi(argv[1]);
    std::string config_file = argv[2];
    ConfigMap config;
    read_config(config_file, config);

    std::string base_filepath        = config["base_filepath"];
    std::string query_filepath       = config["query_filepath"];
    std::string ground_truth_filepath= config["ground_truth_filepath"];
    std::string data_format          = config["data_format"];
    int d = std::stoi(config["d"]);
    int nb = std::stoi(config["nb"]);
    int nq = std::stoi(config["nq"]);
    int k  = std::stoi(config["k"]);
    int partitions = std::stoi(config["partitions"]);
    std::string build_index_store_path     = config["build_index_store_path"];
    std::string build_disk_store_path      = config["build_disk_store_path"];
    std::string build_centroid_index_path  = config["build_centroid_index_path"];
    std::string search_index_store_path    = config["search_index_store_path"];
    std::string search_disk_store_path     = config["search_disk_store_path"];
    std::string search_centroid_index_path = config["search_centroid_index_path"];
    //********************************************build param********************************************************
    
    int nlist = std::stoi(config["nlist"]);
    int m = std::stoi(config["m"]);
    int nbits = std::stoi(config["nbits"]);
    int replicas = std::stoi(config["replicas"]);
    float shrink_replicas = std::stof(config["shrink_replicas"]);
    float build_estimate_factor = std::stof(config["build_estimate_factor"]);
    float build_prune_factor = std::stof(config["build_prune_factor"]);
    std::string build_metric_type = config["build_metric_type"];
    std::string dataset_fmt = config["dataset_fmt"];
    std::string vector_type = config["vector_type"];
    int build_threads = std::stoi(config["build_threads"]);
    int ratio = std::stoi(config["ratio"]);   // sample ratio
        //int top_clusters = config.count("top_clusters") ? std::stoi(config["top_clusters"]) : 5;  // don't actually need, but some function need this

    // *********************************************search param***************************************************
    std::string queryset_fmt = config["queryset_fmt"];
    std::string truthset_fmt = config["truthset_fmt"];

    int k_per_partition  = std::stoi(config["k_per_partition"]);
    float search_estimate_factor         = std::stof(config["search_estimate_factor"]);
    float search_estimate_factor_high_dim= std::stof(config["search_estimate_factor_high_dim"]);
    float search_prune_factor            = std::stof(config["search_prune_factor"]);
    size_t search_top                    = std::stoi(config["search_top"]);

    auto search_nprobes = parse_size_t_list(config["search_nprobes"]);
    auto search_estimate_factors_partial = parse_float_list(config["search_estimate_factors_partial"]);

    int max_continous_pages              = std::stoi(config["max_continous_pages"]);
    int full_decode_volume               = std::stoi(config["full_decode_volume"]);
    int partial_one_decode_volume        = std::stoi(config["partial_one_decode_volume"]);
    int partial_two_decode_volume        = std::stoi(config["partial_two_decode_volume"]);
    int submit_per_round                 = std::stoi(config["submit_per_round"]);
    int search_threads                   = std::stoi(config["search_threads"]);
    int cache_vectors                    = std::stoi(config["cache_vectors"]);
    int query_for_warm_up                = std::stoi(config["query_for_warm_up"]);
    int verbose_num                      = std::stoi(config["verbose"]);

    bool verbose = (verbose_num != 0);

    // -------- helpers ----------
    auto get_s = [&](const std::string& k, const std::string& def)->std::string {
        auto it = config.find(k); return (it==config.end()) ? def : it->second;
    };
    auto get_i = [&](const std::string& k, int def)->int {
        auto it = config.find(k); return (it==config.end()) ? def : std::stoi(it->second);
    };
    auto get_f = [&](const std::string& k, float def)->float {
        auto it = config.find(k); return (it==config.end()) ? def : std::stof(it->second);
    };
    auto has = [&](const std::string& k)->bool { return config.find(k)!=config.end(); };

    int build_memory_graph_efb = get_i("build_memory_graph_efb", 40);
    int build_memory_graph_efs = get_i("build_memory_graph_efs", 16);
    int build_memory_graph_M = get_i("build_memory_graph_M", 16);

    // -------- NEW: read multi-disk loading configs with fallback ----------
    int disk1_count = get_i("disk1_count", 0);  // 0 => no split, backward compatible
    if (disk1_count < 0) disk1_count = 0;
    if (disk1_count > partitions) disk1_count = partitions;

    std::string disk1_store_base    = get_s("search_disk1_disk_store_path",     search_disk_store_path);
    std::string disk1_centroid_base = get_s("search_disk1_centroid_index_path", search_centroid_index_path);
    std::string disk2_store_base    = get_s("search_disk2_disk_store_path",     search_disk_store_path);
    std::string disk2_centroid_base = get_s("search_disk2_centroid_index_path", search_centroid_index_path);

    // -------- NEW: per-disk param sets (used by load_indices; fallback to global) ----------
    float est1   = get_f("search1_estimate_factor",          search_estimate_factor);
    float est_hd1= get_f("search1_estimate_factor_high_dim", search_estimate_factor_high_dim);
    float pr1    = get_f("search1_prune_factor",             search_prune_factor);
    size_t top1  = (size_t)get_i("search1_top",              (int)search_top);
    int reps1    = get_i("search1_replicas",                 replicas);
    int maxp1    = get_i("search1_max_continous_pages",      max_continous_pages);
    int fdv1     = get_i("search1_full_decode_volume",       full_decode_volume);
    int pdv1     = get_i("search1_partial_one_decode_volume",partial_one_decode_volume);
    int ptv1     = get_i("search1_partial_two_decode_volume",partial_two_decode_volume);
    int spr1     = get_i("search1_submit_per_round",         submit_per_round);

    float est2   = get_f("search2_estimate_factor",          search_estimate_factor);
    float est_hd2= get_f("search2_estimate_factor_high_dim", search_estimate_factor_high_dim);
    float pr2    = get_f("search2_prune_factor",             search_prune_factor);
    size_t top2  = (size_t)get_i("search2_top",              (int)search_top);
    int reps2    = get_i("search2_replicas",                 replicas);
    int maxp2    = get_i("search2_max_continous_pages",      max_continous_pages);
    int fdv2     = get_i("search2_full_decode_volume",       full_decode_volume);
    int pdv2     = get_i("search2_partial_one_decode_volume",partial_one_decode_volume);
    int ptv2     = get_i("search2_partial_two_decode_volume",partial_two_decode_volume);
    int spr2     = get_i("search2_submit_per_round",         submit_per_round);

    // -------- NEW: optional per-disk overrides for runtime loop ----------
    // - Provide values to fix per disk; otherwise continue using the loop's nprobes / estimate_factors_partial.
    int   nprobe1_opt  = has("search1_nprobe") ? std::stoi(config["search1_nprobe"]) : -1;
    int   nprobe2_opt  = has("search2_nprobe") ? std::stoi(config["search2_nprobe"]) : -1;
    float partial1_opt = has("search1_estimate_factor_partial") ? std::stof(config["search1_estimate_factor_partial"]) : -1.0f;
    float partial2_opt = has("search2_estimate_factor_partial") ? std::stof(config["search2_estimate_factor_partial"]) : -1.0f;



    if (type == 0) {
        disk_build(base_filepath.c_str(), nb, d, ratio, 
            build_index_store_path, build_disk_store_path, build_centroid_index_path, partitions,
            nlist, m, nbits, replicas, shrink_replicas, build_estimate_factor, build_prune_factor, build_metric_type, dataset_fmt, vector_type,
            build_threads, 3, verbose, build_memory_graph_efb, build_memory_graph_efs, build_memory_graph_M
        );
    } else if (type == 1) {
        disk_search(query_filepath.c_str(), ground_truth_filepath.c_str(), nq, d, k, k_per_partition,
            search_disk_store_path, search_centroid_index_path, partitions,
            search_estimate_factor, search_estimate_factor_high_dim,
            search_prune_factor, search_top, replicas, search_nprobes, search_estimate_factors_partial, 
            max_continous_pages, full_decode_volume, partial_one_decode_volume, partial_two_decode_volume, 
            submit_per_round, queryset_fmt, truthset_fmt, search_threads, cache_vectors, query_for_warm_up, verbose,
            // split bases
            disk1_count, disk1_store_base, disk1_centroid_base, disk2_store_base, disk2_centroid_base,
            // per-disk param sets for loading
            est1, est_hd1, pr1, top1, reps1, maxp1, fdv1, pdv1, ptv1, spr1,
            est2, est_hd2, pr2, top2, reps2, maxp2, fdv2, pdv2, ptv2, spr2,
            // optional per-disk runtime overrides
            nprobe1_opt, nprobe2_opt, partial1_opt, partial2_opt
        );
    }
    return 0;
}
