#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>

#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <string.h>
#include <omp.h>

using idx_t = faiss::idx_t;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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
    fread(&d, sizeof(int), 1, f);
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
    fread(&d, 1, sizeof(int), f);
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
    fread(&n, sizeof(uint32_t), 1, f);
    fread(&d, sizeof(uint32_t), 1, f);

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


void write_file(int* data, const std::string& filename, size_t nq, int k, int dimension) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Write data from the array to disk
        for (size_t i = 0; i < nq; ++i) {
            // Write the value of k
            file.write(reinterpret_cast<const char*>(&k), sizeof(int));
            // TODO: investigate duplicated values during write
            // Write k vector entries
            file.write(reinterpret_cast<const char*>(data + i * k), sizeof(int) * k);
        }
        file.close(); // Close file
    } else {
        std::cerr << "HYH:WRITE ERROR" << std::endl;
    }
}


int main() {
    // dimension of the vectors to index
    int d = 0;
    // size of the database we plan to index
    size_t nb = 1000000;

    size_t dd; // dimension
    size_t nt; // number of vectors

    // std::string base_filepath = "/mnt/d/VectorDB/Deep1M/Deep10M/base.10M.fbin";
    // std::string query_filepath = "/mnt/d/VectorDB/Deep1M/Deep10M/query.public.10K.fbin"; 
    // std::string groundtruth_filepath = "/mnt/d/VectorDB/Deep1M/Deep10M/deep10M_groundtruth.ivecs";


    // std::string base_filepath =        "/mnt/d/VectorDB/Turing/Turing10M/msturing-10M-clustered.fbin";
    // std::string query_filepath =       "/mnt/d/VectorDB/Turing/Turing10M/testQuery10K.fbin"; 
    // std::string groundtruth_filepath = "/mnt/d/VectorDB/Turing/Turing10M/msturing-10M-gt.ivecs";


    std::string base_filepath =        "/mnt/d/VectorDB/Text2Image/T2I1m/base.1M.fbin";
    std::string query_filepath =       "/mnt/d/VectorDB/Text2Image/T2I1m/query.public.100K.fbin"; 
    std::string groundtruth_filepath = "/mnt/d/VectorDB/Text2Image/T2I1m/T2I_gt.ivecs";

    //float* xb = load_and_convert_to_float(base_filepath, &dd, &nt);
    float* xb = fbin_read(base_filepath.c_str(), &dd, &nt);
    d = dd;
    std::cout << "d:" << dd << "  ," << d << "\n";
    float* xq = fbin_read(query_filepath.c_str(), &dd, &nt);
    // Adjust file reading approach

    // generate labels by IndexFlat
    faiss::IndexFlat bruteforce(d, faiss::METRIC_L2);

    bruteforce.add(nb, xb);
    
    // Number of query vectors
    size_t nq = 10000;

    { // searching the database
        printf("Searching ...\n");
        // Generate query vectors
        // k-NN, k nearest neighbors
        int k = 100;
        
        // Brute-force search
        std::vector<faiss::idx_t> gt_nns(k * nq);
        std::vector<int> gt_nns_4B(k * nq);

        //std::vector<int>
        // Store additional data
        std::vector<float> dis(k * nq);
        std::vector<float> dis_gnn(k * nq);
        bruteforce.search(nq, xq, k, dis_gnn.data(), gt_nns.data());
        for (size_t i = 0; i < k * nq; ++i) {
            if (gt_nns[i] > std::numeric_limits<int>::max()) {
                std::cerr << "Warning: Value " << gt_nns[i] << " exceeds the maximum limit of int int." << std::endl;
            }
            gt_nns_4B[i] = static_cast<int>(gt_nns[i]);
            //std::cout << "int:" << gt_nns_4B[i] << "   gt" << gt_nns[i] << std::endl;
         }


        //std::string filename = "D:\\VectorDB\\sift\\sift_groundtruth_2.ivecs";
        //std::string filename = "D:\\VectorDB\\RaBitQ\\RaBitQ\\data\\glove2.2m\\glove2.2m_groundtruth.ivecs";
        //std::string filename = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_groundtruth.ivecs";

        write_file(gt_nns_4B.data(),groundtruth_filepath, nq, k, d);

        std::cout << "write data successfully!" << std::endl;
        
    }
    return 0;
}
