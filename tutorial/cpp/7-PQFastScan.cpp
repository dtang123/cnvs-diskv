/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexRefine.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <omp.h>

using idx_t = faiss::idx_t;
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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
int main() {
    double t0 = elapsed();
    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 10000;   // nb of queries
    char* base_filepath = "/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    char* query_filepath = "/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    char* groundtruth_filepath = "/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    size_t dd; // dimension
    size_t nt; // the number of vectors
    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    size_t dd2; // dimension
    size_t nt2; // the number of query
    xb = fvecs_read(base_filepath, &dd, &nt);
    xq = fvecs_read(query_filepath, &dd2, &nt2);

    {
        srand((int)time(0));
        std::vector<float> trainvecs(nb / 100 * d);
        for (int i = 0; i < nb / 100; i++) {
            int rng = (rand() % (nb + 1));
            ; // rng=random number in nb

            for (int j = 0; j < d; j++) {
                // printf(" setting %d vector's %d data, trianvecs[%d]=xb[%d]
                // \n",i,j,d * i + j,rng * d + j);
                trainvecs[d * i + j] = xb[rng * d + j];
            }
        }
        int nlist = 1000;
        int k = 100;
        // load ground-truth and convert int to long
        size_t nq2;
        size_t kk;
        int* gt_int = ivecs_read(
                groundtruth_filepath,
                &kk,
                &nq2);
        faiss::IndexFlatL2 coarse_quantizer(d);
        // faiss::IndexIVFFlat index(&quantizer, d, nlist);
        // faiss::Index* index;
        // index= faiss::index_factory(128, "IVF1024,PQ64x4fs");
        // faiss::IndexPQFastScan index(d,64,4,faiss::METRIC_L2);
        faiss::IndexIVFPQFastScan index(
                &coarse_quantizer, 128, 1000, 64, 4, faiss::METRIC_L2);
        index.set_assign_replicas(2);
        index.by_residual=false;
        faiss::IndexRefineFlat newindex(&index);
        newindex.k_factor = 3;
        
        // faiss::Index * index1 = faiss::read_index("large.index");

        // assert(!index.is_trained);
        //  faiss::ParameterSpace params;
        //  params.initialize(index);
        printf("[%.3f s] IndexIVFPQFastScan start to train\n", elapsed() - t0);
        newindex.train(nb / 100, trainvecs.data());
        printf("[%.3f s] IndexIVFPQFastScan train finished\n", elapsed() - t0);
        // assert(index.is_trained);
        printf("[%.3f s] IndexIVFPQFastScan start to add\n", elapsed() - t0);
        newindex.add(nb, xb);
        printf("[%.3f s] IndexIVFPQFastScan add finished\n", elapsed() - t0);
        std::vector<double> search_times;
        std::vector<double> recalls;
        omp_set_num_threads(1);
        { // search xq

            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            faiss::idx_t* gt;
            gt = new faiss::idx_t[kk * nq];
            for (int i = 0; i < kk * nq; i++) {
                gt[i] = gt_int[i]; // long int / int
            }
            delete[] gt_int;
            // printf("I=\n");
            // for (int i = nq - 5; i < nq; i++) {
            //     for (int j = 0; j < k; j++)
            //         printf("%5zd ", I[i * k + j]);
            //     printf("\n");
            // }
            int arr[] = {5, 7, 10, 20, 30, 40};

            // write_index(index, "hnsw.index");
            // how to get length of array

            for (int i : arr) {
                // faiss::SearchParametersIVF params;
                // print current i
                printf("-----------now, the efs is=%d---------\n", i);
                // index.hnsw.efSearch = i;
                index.nprobe = i;
                // params.nprobe=i;
                printf("[%.3f s] IndexIVFPQFastScan start to search\n",
                       elapsed() - t0);
                // newindex.search(nq, xq, k, D, I);
                                double t1 = elapsed();
                newindex.search(nq, xq, k, D, I);
                double t2 = elapsed();
                double search_time = t2 - t1;
                // iterative query
                //  for (int i = 0; i < nq; i++) {
                //      index->search(1, xq + i * d, k, D + i * k, I + i * k);
                //  }
                printf("[%.3f s] IndexIVFPQFastScan search finished\n",
                       elapsed() - t0);

                // evaluate result
                int n2_100 = 0;
                for (int i = 0; i < nq; i++) {
                    std::map<float, int> umap;
                    for (int j = 0; j < k; j++) {
                        umap.insert({gt[i * k + j], 0});
                    }
                    for (int l = 0; l < k; l++) {
                        if (umap.find(I[i * k + l]) != umap.end()) {
                            n2_100++;
                        }
                    }
                    umap.clear();
                }
                printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k));
                double recall = n2_100 / float(nq * k);
                search_times.push_back(search_time/nq*1000);
                recalls.push_back(recall);
            }

            delete[] I;
            delete[] D;
        }

        delete[] xb;
        delete[] xq;
        // Print the search times and recalls
        std::cout << "Search times: [";
        for (size_t i = 0; i < search_times.size(); i++) {
            std::cout << search_times[i];
            if (i < search_times.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "QPS: [";
        for (size_t i = 0; i < recalls.size(); i++) {
            std::cout << 1000.0/search_times[i];
            if (i < recalls.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "Recalls: [";
        for (size_t i = 0; i < recalls.size(); i++) {
            std::cout << recalls[i];
            if (i < recalls.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        return 0;
    }
}
