#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <string>
#include <iomanip>
#include <fstream>
#include <map>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <omp.h>

#include <aws/core/Aws.h>

#include "faiss/IndexDiskV.h"
#include "faiss/utils/utils.h"
#include <faiss/index_io.h>
#include <faiss/impl/S3IOStats.h>
#include <faiss/impl/S3IOReader.h>

typedef std::map<std::string, std::string> ConfigMap;

void read_config(const std::string& path, ConfigMap& config) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Failed to open config: " << path << "\n"; exit(1); }
    std::string line;
    while (std::getline(in, line)) {
        auto sharp = line.find('#');
        if (sharp != std::string::npos) line = line.substr(0, sharp);
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty()) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        config[line.substr(0, pos)] = line.substr(pos + 1);
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

// ── Loaders ───────────────────────────────────────────────────────────────────
float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", fname); abort(); }
    uint32_t n, d;
    fread(&n, sizeof(uint32_t), 1, f);
    fread(&d, sizeof(uint32_t), 1, f);
    size_t total = (size_t)n * d;
    float* x = new float[total];
    size_t chunk = 1024 * d;
    float* buf = new float[chunk];
    size_t read = 0;
    while (read < total) {
        size_t toread = std::min(chunk, total - read);
        fread(buf, sizeof(float), toread, f);
        std::copy(buf, buf + toread, x + read);
        read += toread;
    }
    delete[] buf;
    fclose(f);
    *d_out = d; *n_out = n;
    return x;
}

int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fbin_read(fname, d_out, n_out);
}

float* load_dataset_vectors(const std::string& path, const std::string& fmt,
                             size_t* d, size_t* n) {
    if (fmt == "fbin") return fbin_read(path.c_str(), d, n);
    std::cerr << "Unsupported format: " << fmt << "\n"; exit(1);
}

int* load_groundtruth(const std::string& path, const std::string& fmt,
                       size_t* d, size_t* n) {
    if (fmt == "ibin") return ibin_read(path.c_str(), d, n);
    std::cerr << "Unsupported gt format: " << fmt << "\n"; exit(1);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
void print_index_info(size_t k, size_t nq, bool clear_stats = true) {
    const double inv = nq ? 1.0 / (double)nq : 0.0;
    auto& s = faiss::indexDiskV_stats;
    std::cout << "  Scanned lists/q     : " << s.searched_lists * inv << "\n"
              << "  disk_full_ms/q      : " << s.disk_full_elapsed.count()/1000.0*inv << "\n"
              << "  disk_partial_ms/q   : " << s.disk_partial_elapsed.count()/1000.0*inv << "\n"
              << "  rank_ms/q           : " << s.rank_elapsed.count()/1000.0*inv << "\n"
              << "  pq_ms/q             : " << s.pq_elapsed.count()/1000.0*inv << "\n"
              << "  vectors_full/q      : " << s.searched_vector_full * inv << "\n"
              << "  vectors_partial/q   : " << s.searched_vector_partial * inv << "\n"
              << "  requests_full/q     : " << s.requests_full * inv << "\n"
              << "  requests_partial/q  : " << s.requests_partial * inv << "\n"
              << "  S3 requests         : " << faiss::s3_io_stats.total_requests.load() << "\n"
              << "  S3 MB read          : "
              << faiss::s3_io_stats.total_bytes_read.load()/(1024.0*1024.0) << "\n";
    if (clear_stats) { faiss::indexDiskV_stats.reset(); faiss::s3_io_stats.reset(); }
}

// ── Load index — exactly mirrors demo_script load_indices ─────────────────────
void load_indices(faiss::IndexDiskV*& index,
                  const std::string& disk_store_path,
                  const std::string& centroid_index_path,
		  const std::string& index_path,
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
                  bool verbose)
{
    std::string index_meta_data_path = index_path  + ".index";
    std::string index_vector_path    = disk_store_path + ".clustered";

    std::cout << "Loading index from: " << index_meta_data_path << "\n";
    if (index_meta_data_path.rfind("s3://", 0) == 0) {
        faiss::S3IOReader reader(index_meta_data_path);
        index = dynamic_cast<faiss::IndexDiskV*>(faiss::read_index(&reader));
    } else {
        index = dynamic_cast<faiss::IndexDiskV*>(
            faiss::read_index(index_meta_data_path.c_str()));
    }
    if (!index) { std::cerr << "ERROR: not an IndexDiskV\n"; exit(1); }

    index->select_lists_path = index_vector_path;
    index->disk_path         = index_vector_path;
    index->set_centroid_index_path(centroid_index_path);
    index->load_hnsw_centroid_index();
    index->set_top(top);
    index->set_assign_replicas(replicas);
    index->set_estimate_factor(estimate_factor);
    index->set_estimate_factor_partial(estimate_factor_partial);
    index->set_estimate_factor_high_dim(estimate_factor_high_dim);
    index->set_prune_factor(prune_factor);
    index->set_max_continous_pages(max_continous_pages);
    index->set_ahead_pq_volumes(full_decode_volume,
                                partial_one_decode_volume,
                                partial_two_decode_volume);
    index->set_submit_per_round(submit_per_round);

    if (verbose) {
        std::cout << "  top=" << top << " replicas=" << replicas
                  << " ef=" << estimate_factor
                  << " ef_partial=" << estimate_factor_partial
                  << " ef_high=" << estimate_factor_high_dim
                  << " prune=" << prune_factor << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: ./run_diskv_search <config.sh>\n";
        return 1;
    }

    Aws::SDKOptions aws_opts;
    Aws::InitAPI(aws_opts);

    ConfigMap config;
    read_config(argv[1], config);

    // ── Parse config ──────────────────────────────────────────────────────────
    std::string query_filepath        = config["query_filepath"];
    std::string ground_truth_filepath = config["ground_truth_filepath"];
    std::string queryset_fmt          = config["queryset_fmt"];
    std::string truthset_fmt          = config["truthset_fmt"];

    int    partitions        = std::stoi(config["partitions"]);
    int    nq                = std::stoi(config["nq"]);
    int    k                 = std::stoi(config["k"]);
    int    k_per_partition   = std::stoi(config["k_per_partition"]);
    int    search_threads    = std::stoi(config["search_threads"]);
    int    cache_vectors     = std::stoi(config["cache_vectors"]);
    int    query_for_warm_up = std::stoi(config["query_for_warm_up"]);
    bool   verbose           = std::stoi(config["verbose"]) != 0;

    float  est_factor        = std::stof(config["search_estimate_factor"]);
    float  est_high_dim      = std::stof(config["search_estimate_factor_high_dim"]);
    float  prune_factor      = std::stof(config["search_prune_factor"]);
    size_t top               = std::stoi(config["search_top"]);
    int    replicas          = std::stoi(config["replicas"]);
    int    max_cont_pages    = std::stoi(config["max_continous_pages"]);
    int    full_vol          = std::stoi(config["full_decode_volume"]);
    int    partial1_vol      = std::stoi(config["partial_one_decode_volume"]);
    int    partial2_vol      = std::stoi(config["partial_two_decode_volume"]);
    int    submit_per_round  = std::stoi(config["submit_per_round"]);

    auto nprobes         = parse_size_t_list(config["search_nprobes"]);
    auto partial_factors = parse_float_list(config["search_estimate_factors_partial"]);

    // ── Load all partition indices ────────────────────────────────────────────
    std::vector<faiss::IndexDiskV*> indices(partitions);
    for (int i = 0; i < partitions; i++) {
        std::string disk_path_i =
            config["search_disk_store_path"] + "_" + std::to_string(i);
        std::string centroid_path_i =
            config["search_centroid_index_path"] + "_" + std::to_string(i);
	std::string index_path_i =
	    config["search_index_meta_path"] + "_" + std::to_string(i);	
	load_indices(indices[i], disk_path_i, centroid_path_i, index_path_i,
                     est_factor, partial_factors[0], est_high_dim,
                     prune_factor, top, replicas, max_cont_pages,
                     full_vol, partial1_vol, partial2_vol,
                     submit_per_round, verbose);
        std::string local_index_file = index_path_i + ".index"; 
        
        // We use a temporary index to steal its InvertedLists
        faiss::Index* temp_index = faiss::read_index(local_index_file.c_str());
        faiss::IndexIVF* temp_ivf = dynamic_cast<faiss::IndexIVF*>(temp_index);
        
        if (temp_ivf && temp_ivf->invlists) {
            // Transfer the InvertedLists metadata to our IndexDiskV
            // This ensures indices[i]->invlists->list_size() is no longer 0
            indices[i]->replace_invlists(temp_ivf->invlists, true);
            temp_ivf->invlists = nullptr; // prevent double-free
        }
        delete temp_index;
    
        // 3. Setup S3
        // Now that the index "knows" how big each list is, 
        // the S3 processor knows how much data to request.
        indices[i]->initializeDiskIO(search_threads);
    }

    std::cout << "Loaded " << partitions << " partition(s)\n"
              << "ntotal=" << indices[0]->ntotal
              << " nlist=" << indices[0]->nlist
              << " d=" << indices[0]->d << "\n";
    std::cout << "valueType=" << indices[0]->valueType << "\n";
    std::cout << "DEBUG index params:\n"
          << "  assign_replicas=" << indices[0]->assign_replicas << "\n"
          << "  top=" << indices[0]->top << "\n"
          << "  estimate_factor=" << indices[0]->estimate_factor << "\n"
          << "  estimate_factor_partial=" << indices[0]->estimate_factor_partial << "\n"
          << "  prune_factor=" << indices[0]->prune_factor << "\n"
          << "  aligned_cluster_info=" << indices[0]->aligned_cluster_info << "\n"
          << "  aligned_inv_info=" << indices[0]->aligned_inv_info << "\n"
          << "  cached_list_info=" << indices[0]->cached_list_info << "\n";
    // Add after load_indices in main():
    std::cout << "[INVLIST CHECK] First 5 IDs in list 0:\n";
    const faiss::idx_t* list0_ids = indices[0]->invlists->get_ids(0);
    size_t list0_size = indices[0]->invlists->list_size(0);
    std::cout << "  list_size=" << list0_size << "\n";
    for (int j = 0; j < std::min((size_t)5, list0_size); j++)
        std::cout << "  id[" << j << "]=" << list0_ids[j] << "\n";
    std::cout << "[INV_INFO CHECK] aligned_inv_info[0]:\n"
          << "  page_start=" << indices[0]->aligned_inv_info[0].page_start << "\n"
          << "  page_count=" << indices[0]->aligned_inv_info[0].page_count << "\n"
          << "  list_size=" << indices[0]->aligned_inv_info[0].list_size << "\n";
    // ── Load queries and ground truth ─────────────────────────────────────────
    size_t dd, nq_file, kk, nq2;
    float* xq     = load_dataset_vectors(query_filepath, queryset_fmt, &dd, &nq_file);
    int*   gt_int = load_groundtruth(ground_truth_filepath, truthset_fmt, &kk, &nq2);
    nq = std::min(nq, (int)nq_file);

    // Convert gt to idx_t like demo_script
    faiss::idx_t* gt = new faiss::idx_t[k * nq];
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < k; j++)
            gt[i*k+j] = static_cast<faiss::idx_t>(gt_int[i*(int)kk+j]);
    delete[] gt_int;

    std::cout << "Loaded " << nq << " queries dim=" << dd
              << ", GT k=" << kk << "\n";

    // ── Warm-up all partitions — mirrors demo_script exactly ─────────────────
    omp_set_num_threads(search_threads);
    for (int i = 0; i < partitions; i++) {
        printf("Warming up partition %d...\n", i);
	std::string original_disk_path = indices[i]->disk_path;
        std::cout << "  disk_path=" << indices[i]->disk_path << "\n";
    
        bool is_s3 = (indices[i]->disk_path.rfind("s3://", 0) == 0);
        
        // Skip warmUpListCache for S3 (opens disk_path with ifstream)
        if (!is_s3) {
            indices[i]->warmUpListCache(1000, xq, 200, 0);
        }
 //	indices[i]->disk_path = config["search_index_meta_path"] + "_" + std::to_string(i) + ".index";
//	indices[i]->warmUpListCache(1000, xq, 200, 0);
   //     indices[i]->disk_path = config["search_index_meta_path"] + "_" + std::to_string(i) + ".clustered";
	indices[i]->warmUpAllIndexMetadata();
	indices[i]->disk_path = original_disk_path;
        indices[i]->initializeDiskIO(search_threads);

        if (cache_vectors > 0) {
            size_t n_shard = 100000;
            indices[i]->set_cache_strategy(faiss::IMMEDIATELY_UPDATE);
            size_t warmed = indices[i]->warmUpVectorCacheShard(
                query_for_warm_up, xq, 100, 100, cache_vectors,
                partial_factors[0], search_threads, n_shard);
            std::cout << "  Partition " << i << " vector cache: "
                      << warmed << " vectors\n";
            indices[i]->shutdownDiskIO(search_threads);
        }
    }

    // ── Per-partition result buffers ──────────────────────────────────────────
    std::vector<std::vector<faiss::idx_t>> Is(partitions,
        std::vector<faiss::idx_t>(k_per_partition * nq));
    std::vector<std::vector<float>> Ds(partitions,
        std::vector<float>(k_per_partition * nq));

    // ── Sweep nprobes x partial_factors — mirrors demo_script ────────────────
    std::vector<double> search_times;
    std::vector<double> recalls;
    faiss::indexDiskV_stats.reset();
    faiss::s3_io_stats.reset();

    bool multi_filter = (partitions > 1);

    std::cout << "\n════════════════════════════════════════════════\n";
    for (size_t nprobe : nprobes) {
        for (float ep : partial_factors) {

            // Set params on all partitions
            for (int i = 0; i < partitions; i++) {
                indices[i]->nprobe = nprobe;
                indices[i]->set_estimate_factor_partial(ep);
            }

            double search_time = 0;

            // Search each partition — mirrors demo_script partition loop
            for (int i = 0; i < partitions; i++) {
                indices[i]->initializeDiskIO(search_threads);

                // Multi-filter: use previous partition's worst result as threshold
                if (multi_filter && i != 0) {
                    std::vector<float> thresholds(nq);
                    for (int qi = 0; qi < nq; qi++)
                        thresholds[qi] = Ds[i-1][(qi+1)*k_per_partition - 1];
		    indices[i]->set_search_threshold(nq, thresholds);
		    indices[i]->q_indicator = new size_t[search_threads];
                }

                auto t0 = std::chrono::high_resolution_clock::now();
		std::cout << "Starting index search\n";
		indices[i]->search(nq, xq, k_per_partition,
                                   Ds[i].data(), Is[i].data());
                auto t1 = std::chrono::high_resolution_clock::now();
                search_time += std::chrono::duration<double>(t1 - t0).count();

                indices[i]->shutdownDiskIO(search_threads);

                // Cleanup multi-filter state
                indices[i]->clear_search_threshold();
                if (indices[i]->q_indicator != nullptr) {
                    delete[] indices[i]->q_indicator;
                    indices[i]->q_indicator = nullptr;
                }
            }

            // ── Merge results across partitions — mirrors demo_script ─────────
            int correct = 0;
	    std::cout << "[RECALL DEBUG]\n";
            std::cout << "  nq=" << nq << " k=" << k
                      << " k_per_partition=" << k_per_partition << "\n";
            std::cout << "  GT first query first 5 IDs: ";
            for (int j = 0; j < 5; j++) std::cout << gt[j] << " ";
            std::cout << "\n  Result first query first 5 IDs: ";
            for (int j = 0; j < 5; j++) std::cout << Is[0][j] << " ";
            std::cout << "\n  Result first query first 5 dists: ";
            for (int j = 0; j < 5; j++) std::cout << Ds[0][j] << " ";
            std::cout << "\n";
            for (int qi = 0; qi < nq; qi++) {
                // Collect all results from all partitions
                std::vector<std::pair<float, faiss::idx_t>> all_results;
                all_results.reserve(partitions * k_per_partition);
                for (int pi = 0; pi < partitions; pi++) {
                    for (int j = 0; j < k_per_partition; j++) {
                        all_results.push_back({
                            Ds[pi][qi * k_per_partition + j],
                            Is[pi][qi * k_per_partition + j]
                        });
                    }
                }
                // Sort by distance, take top k
                std::sort(all_results.begin(), all_results.end());
                int top_k = std::min(k, (int)all_results.size());

                // Build GT set for this query
                std::unordered_map<faiss::idx_t, int> gt_set;
                for (int j = 0; j < k; j++)
                    gt_set[gt[qi * k + j]] = 0;

                // Count recall hits
                for (int j = 0; j < top_k; j++) {
                    if (gt_set.count(all_results[j].second))
                        correct++;
                }
		if (qi == 0) {
                    printf("\n=== First Query Debug ===\n");
                    printf("GT ids (first 10): ");
                    for (int j = 0; j < std::min(10, k); j++)
                        printf("%ld ", gt[j]);
                    printf("\nResult ids (first 10): ");
                    for (int j = 0; j < std::min(10, (int)all_results.size()); j++)
                        printf("%ld(d=%.3f) ", all_results[j].second, all_results[j].first);
                    printf("\ngt_set size: %zu\n", gt_set.size());
                    printf("top_k: %d\n", top_k);
                    printf("===\n\n");
                }
            }

            double recall  = correct / (double)(nq * k);
            double qps     = nq / search_time;

            printf("nprobe=%zu ep=%.3f | QPS=%.1f | R@%d=%.4f | time=%.3fs\n",
                   nprobe, ep, qps, k, recall);

            if (verbose) print_index_info(k, nq, false);

            search_times.push_back(search_time / nq * 1000.0);
            recalls.push_back(recall);

            faiss::indexDiskV_stats.reset();
            faiss::s3_io_stats.reset();
        }
    }

    // ── Summary output matching demo_script format ────────────────────────────
    std::cout << "\nSearch times: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << search_times[i];
        if (i < search_times.size()-1) std::cout << ", ";
    }
    std::cout << "]\nQPS         : [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << (1000.0 / search_times[i]);
        if (i < search_times.size()-1) std::cout << ", ";
    }
    std::cout << "]\nRecalls     : [";
    for (size_t i = 0; i < recalls.size(); i++) {
        std::cout << recalls[i];
        if (i < recalls.size()-1) std::cout << ", ";
    }
    std::cout << "]\nTime Gap    : [";
    for (size_t i = 1; i < search_times.size(); i++) {
        std::cout << search_times[i] - search_times[i-1];
        if (i < search_times.size()-1) std::cout << ", ";
    }
    std::cout << "]\n════════════════════════════════════════════════\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] xq;
    delete[] gt;
    for (int i = 0; i < partitions; i++) delete indices[i];
    Aws::ShutdownAPI(aws_opts);
    return 0;
}
