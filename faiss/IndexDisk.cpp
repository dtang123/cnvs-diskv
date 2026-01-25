
#include <faiss/IndexDisk.h>

#include <vector>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <algorithm>
#include <memory>

#include <sys/mman.h>    // mmap, munmap
#include <fcntl.h>       
#include <sys/stat.h>    
#include <unistd.h>      
#include <stdexcept>     

#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>

// Function to save data to a new file
void save_data(
        const float* data,
        const std::string& data_file_path,
        size_t n,
        size_t d) {
    std::ofstream out(data_file_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), n * d * sizeof(float));
    out.close();
}



namespace faiss {

IndexDisk::IndexDisk(
    size_t d,
    const std::string& diskPath,
    MetricType metric
) : Index(d, metric), disk_path(diskPath + ".processed") {}

IndexDisk::IndexDisk() {}

IndexDisk::~IndexDisk() {
// maybe do nothing
}

void IndexDisk::add(idx_t n, const float* x) {
    this->ntotal = n;
    // If we don't have processed data (without indicators), process it
    save_data(x, disk_path, n, d);
}

void IndexDisk::train(idx_t n, const float* x) {
    // do nothing
}

void IndexDisk::reset() {
    // Do nothing
}

void IndexDisk::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in) const{
                // Do nothing
            }

//#define USE_IFSTREAM
//#define USE_MMAP
#define USE_FREAD


// DiskDistanceComputer implementation
struct DiskDistanceComputer : DistanceComputer {
    size_t d;
    const float* q = nullptr;
    size_t code_size; // Size of each code/vector

#ifdef USE_IFSTREAM 
    std::ifstream disk_data;
#endif

#ifdef USE_MMAP        
    int disk_fd = -1;           // file desciptor
    size_t file_size = 0; 
    void* disk_data = nullptr; // mapped_data
#endif

#ifdef USE_FREAD
    FILE* disk_data = nullptr;  // FILE pointer for fread
#endif

    DiskDistanceComputer(const std::string& disk_path, size_t d):d(d), code_size(d*sizeof(float)) {
        
#ifdef USE_IFSTREAM        
        disk_data.open(disk_path, std::ios::binary);
#endif

#ifdef USE_MMAP        
        disk_fd = open(disk_path.c_str(), O_RDONLY);
        struct stat sb;
        fstat(disk_fd, &sb);
        file_size = sb.st_size;
        disk_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, disk_fd, 0);
#endif

#ifdef USE_FREAD
        disk_data = fopen(disk_path.c_str(), "rb");
        if (!disk_data) {
            throw std::runtime_error("DiskDistanceComputer: Failed to open disk file for reading");
        }
#endif

    }

    // Default constructor (not used in this case)
    DiskDistanceComputer() {}

    void set_query(const float* x) override {
        q = x;
    } 

    float operator()(idx_t i) override {
        
        auto time_start = std::chrono::high_resolution_clock::now();      // time begin

        std::vector<float> buffer(d);
#ifdef USE_IFSTREAM
        disk_data.seekg(i * code_size, std::ios::beg);
        disk_data.read(reinterpret_cast<char*>(buffer.data()), d * sizeof(float));
#endif
#ifdef USE_MMAP
        memcpy(buffer.data(), static_cast<char*>(disk_data) + i * code_size, code_size);
#endif 
#ifdef USE_FREAD
        fseek(disk_data, i * code_size, SEEK_SET);
        fread(buffer.data(), sizeof(float), d, disk_data);
#endif

        auto time_end = std::chrono::high_resolution_clock::now();       // time end
        indexDisk_stats.disk_elapsed += time_end - time_start;

        time_start = std::chrono::high_resolution_clock::now();      // time begin

        float dis = fvec_L2sqr(q, buffer.data(), d);

        time_end = std::chrono::high_resolution_clock::now();       // time end
        indexDisk_stats.memory_elapsed += time_end - time_start;
        indexDisk_stats.rerank++;
        return dis;
    }

    float symmetric_dis(idx_t i, idx_t j) {
        std::vector<float> buffer(2 * d);
#ifdef USE_IFSTREAM
        disk_data.seekg(i * code_size, std::ios::beg);
        disk_data.read(reinterpret_cast<char*>(buffer.data()), code_size);
        disk_data.seekg(j * code_size, std::ios::beg);
        disk_data.read(reinterpret_cast<char*>(buffer.data() + d), code_size);
#endif

#ifdef USE_MMAP
        memcpy(buffer.data(), static_cast<char*>(disk_data) + i * code_size, code_size);
        memcpy(buffer.data() + d, static_cast<char*>(disk_data) + j * code_size, code_size);
#endif

#ifdef USE_FREAD
        fseek(disk_data, i * code_size, SEEK_SET);
        fread(buffer.data(), sizeof(float), d, disk_data);
        fseek(disk_data, j * code_size, SEEK_SET);
        fread(buffer.data() + d, sizeof(float), d, disk_data);
#endif

        return fvec_L2sqr(buffer.data(), buffer.data()+d, d);
    }

    
    // Destructor
    ~DiskDistanceComputer() {
#ifdef USE_IFSTREAM
        if (disk_data.is_open()) {
            disk_data.close();
        }
#endif

#ifdef USE_MMAP
    if (disk_data && disk_data != MAP_FAILED) {
            munmap(disk_data, file_size);
        }
    if (disk_fd != -1) {
        close(disk_fd);
    }
#endif

#ifdef USE_FREAD
    if (disk_data) {
        fclose(disk_data);
    }
#endif
    }
};


DistanceComputer* IndexDisk::get_distance_computer() const {
    return new DiskDistanceComputer(disk_path, d);
}

IndexDiskStats indexDisk_stats;

void IndexDiskStats::reset() {
    memset(this, 0, sizeof(*this));
}

} // namespace faiss

