#ifndef FAISS_INDEX_Disk_H
#define FAISS_INDEX_Disk_H

#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>

#include <string>
#include <fstream>   

#include <chrono>
using namespace std::chrono;

namespace faiss {

class IndexDisk : public Index {
public:
    
    IndexDisk(
            //Index* quantizer,
            size_t d,
            const std::string& diskPath,
            MetricType metric = METRIC_L2);

    IndexDisk();

    ~IndexDisk();

    // do nothing
    void add(idx_t n, const float* x) override;

    // do nothing
    void train(idx_t n, const float* x) override;

    // maybe do nothing
    void reset() override;

    DistanceComputer* get_distance_computer() const override;

    void set_disk_read(const std::string& diskPath); // Method to set the disk path and open read stream

    const std::string& get_disk_path() const {
        return this->disk_path;
    }
    
    int get_dim() const {
        return this->d;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in = nullptr ) const override;

    std::string disk_path;
};


struct IndexDiskStats {

    size_t rerank;

    std::chrono::duration<double, std::micro> disk_elapsed;
    std::chrono::duration<double, std::micro> memory_elapsed;

    IndexDiskStats() {
        reset();
    }
    void reset();
};

// global var that collects them all
FAISS_API extern IndexDiskStats indexDisk_stats;

} // namespace faiss

#endif