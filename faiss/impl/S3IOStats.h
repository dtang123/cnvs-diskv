#pragma once
#include <atomic>
#include <iostream>
#include <iomanip>

namespace faiss {

struct S3IOStats {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_bytes_read{0};
    std::atomic<uint64_t> partial_requests{0};
    std::atomic<uint64_t> full_requests{0};
    std::atomic<uint64_t> info_requests{0};

    void reset() {
        total_requests   = 0;
        total_bytes_read = 0;
        partial_requests = 0;
        full_requests    = 0;
        info_requests    = 0;
    }

    void print(double elapsed_sec, size_t nq) const {
        double gb_read     = total_bytes_read.load() / (1024.0 * 1024.0 * 1024.0);
        double mb_per_query = (total_bytes_read.load() / (1024.0 * 1024.0)) / nq;
        std::cout << "\n══════════════════════════════════════════════\n";
        std::cout << "  S3 / MinIO I/O Stats\n";
        std::cout << "──────────────────────────────────────────────\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Total S3 requests  : " << total_requests.load()   << "\n";
        std::cout << "  Full  (top) GETs   : " << full_requests.load()    << "\n";
        std::cout << "  Partial GETs       : " << partial_requests.load() << "\n";
        std::cout << "  Info GETs          : " << info_requests.load()    << "\n";
        std::cout << "  Total data read    : " << gb_read << " GB\n";
        std::cout << "  Data / query       : " << mb_per_query << " MB\n";
        std::cout << "  Requests / query   : "
                  << (double)total_requests.load() / nq << "\n";
        std::cout << "  Throughput         : "
                  << (gb_read / elapsed_sec) << " GB/s\n";
        std::cout << "══════════════════════════════════════════════\n";
    }
};

// Single global instance — defined in IndexDiskV.cpp, declared here
extern S3IOStats s3_io_stats;

} // namespace faiss
