#pragma once
#include <unistd.h>

#define PAGE_SIZE 4096

namespace faiss{

struct Aligned_Cluster_Info{
    size_t page_start;    // 1. begining site of the cluster
    size_t padding_offset;   // 2. padding offset of the cluster
    size_t page_count;    // 3. page usage of the cluster
};

struct Aligned_Invlist_Info : Aligned_Cluster_Info{
    size_t list_size;
};
}

// Use it in search_partially
// identify a vector's position by pages
struct Page_to_Search{
    int first;
    int last;
    int count_pages(){
        return last - first + 1;
    }
};