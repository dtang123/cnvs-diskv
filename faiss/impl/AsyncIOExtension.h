#ifndef FAISS_ASYNC_IO_EXTENSION_H
#define FAISS_ASYNC_IO_EXTENSION_H


#include <stdint.h>
#include <vector>
#include <string.h>
#include <faiss/MetricType.h>
#include <functional>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/DiskIOStructure.h>

//#include <linux/aio_abi.h>
//#define PAGE_SIZE 4096

#if (__cplusplus < 201703L)
#define ALIGN_ALLOC(size) _mm_malloc(size, 32)
#define ALIGN_FREE(ptr) _mm_free(ptr)
#define PAGE_ALLOC(size) _mm_malloc(size, 512)
#define PAGE_FREE(ptr) _mm_free(ptr)
#else
#define ALIGN_ALLOC(size) ::operator new(size, (std::align_val_t)32)
#define ALIGN_FREE(ptr) ::operator delete(ptr, (std::align_val_t)32)
#define PAGE_ALLOC(size) ::operator new(size, (std::align_val_t)512)
#define PAGE_FREE(ptr) ::operator delete(ptr, (std::align_val_t)512)
#endif

namespace faiss{

template<typename T>
class PageBuffer
{
public:
    PageBuffer()
        : m_pageBufferSize(0)
    {
    }

    void ReservePageBuffer(std::size_t p_size)
    {
        if (m_pageBufferSize < p_size)
        {
            m_pageBufferSize = p_size;
            m_pageBuffer.reset(static_cast<T*>(PAGE_ALLOC(sizeof(T) * m_pageBufferSize)), [=](T* ptr) { PAGE_FREE(ptr); });
        }
    }

    T* GetBuffer()
    {
        return m_pageBuffer.get();
    }

    std::size_t GetPageSize()
    {
        return m_pageBufferSize;
    }

private:
    std::shared_ptr<T> m_pageBuffer;

    std::size_t m_pageBufferSize;
};



struct AsyncRequest_Full{
	size_t page_num;     
        
	size_t vectors_num;   
	size_t begin_idx;     
	size_t iobuffer_offset;
	
	const size_t* map;
	const idx_t* ids;
	std::vector<float> pq_dis;

    // use in callback, m_buffer is for aio
    char* m_buffer;
    float* converted_buffer;
	
    AsyncRequest_Full(){
        this->map = nullptr;
        this->ids = nullptr;
    }
	
	AsyncRequest_Full(size_t page_num, size_t vectors_num, size_t begin_idx, size_t list_size, size_t iobuffer_offset,
				const size_t* map, const idx_t* ids, const float* pq_dis):
		page_num(page_num),  vectors_num(vectors_num), begin_idx(begin_idx), iobuffer_offset(iobuffer_offset),
		map(map), ids(ids){
			this->pq_dis.resize(list_size);
			memcpy(this->pq_dis.data(), pq_dis, list_size * sizeof(float));
		}
	
};

struct AsyncRequest_Full_Batch{
	
    std::vector<AsyncRequest_Full> request_full;
	std::uint64_t m_offset;
    std::uint64_t m_readSize;
	size_t total_page_num;
	size_t list_num;

    AsyncRequest_Full_Batch(size_t begin_offset, 
							size_t page_num, 
							size_t vectors_num, 
							size_t begin_idx,
							size_t list_size,
							size_t iobuffer_offset,
							const size_t* map, 
							const idx_t* ids, 
							const float* pq_dis,
							size_t list_num = 1) {
        this->m_offset = begin_offset;
		this->total_page_num = page_num;
		this->m_readSize = page_num * PAGE_SIZE;


        this->list_num = list_num;  
        request_full.reserve(list_num);
		request_full.emplace_back(page_num, vectors_num, begin_idx, list_size, iobuffer_offset, map, ids, pq_dis);
    }

    void push_request(
        size_t page_num, 
        size_t vectors_num, 
		size_t begin_idx,
		size_t list_size,
		size_t iobuffer_offset,
        const size_t* map, 
        const idx_t* ids, 
        const float* pq_dis) {
			this->list_num++;
            this->total_page_num += page_num;
			this->m_readSize += page_num* PAGE_SIZE;
            request_full.emplace_back(page_num, vectors_num, begin_idx, list_size, iobuffer_offset, map, ids, pq_dis);
    }

    const float* get_pq_dis(size_t order = 0){
		return request_full[order].pq_dis.data();
	}
	
    const size_t* get_map(size_t order = 0){
		return request_full[order].map;
	}

};

struct AsyncReadRequests_Full_PQDecode{
	std::vector<AsyncRequest_Full_Batch> list_requests;
	std::vector<PageBuffer<uint8_t>> page_buffers;

	std::function<void()> pq_callback;
	std::function<void(AsyncRequest_Full* requested, void* buffer)> cal_callback;

	std::function<void()> dp_callback;

	void fill_buffer(){
		size_t offset;
		for(int i = 0; i < list_requests.size(); i++){
			offset = 0;
			for(int j = 0; j < list_requests[i].list_num; j++){
				if(j!=0){
					offset += list_requests[i].request_full[j-1].page_num * PAGE_SIZE; 
				}
				list_requests[i].request_full[j].m_buffer = (char*)(page_buffers[i].GetBuffer()) + offset;
			}
		}
	}

};



struct AsyncRequest_Partial{
	size_t page_num;      //pages need to read    
	size_t vectors_num;   
	size_t begin_idx;     // block search
	size_t iobuffer_offset;  // block search
	size_t list_no;

	std::uint64_t m_offset;
    std::uint64_t m_readSize;
	
	const size_t* map;
	const idx_t* ids;

	std::vector<int> in_buffer_offsets;
	std::vector<size_t> in_buffer_ids;

	char* m_buffer;
    float* converted_buffer;

	AsyncRequest_Partial() 
        : page_num(0), vectors_num(0), begin_idx(0), m_offset(0), m_readSize(0), iobuffer_offset(0),
          map(nullptr), ids(nullptr), m_buffer(nullptr), converted_buffer(nullptr) {}

	AsyncRequest_Partial(
        size_t page_num, size_t vectors_num, size_t begin_idx, size_t m_offset, size_t iobuffer_offset,
        const size_t* map, const idx_t* ids,
        const int* in_offsets_start, const int* in_offsets_end,
        const size_t* in_ids_start, const size_t* in_ids_end, size_t list_no = 0
    )
        : page_num(page_num), vectors_num(vectors_num), begin_idx(begin_idx), m_offset(m_offset), iobuffer_offset(iobuffer_offset),
          m_readSize(page_num * PAGE_SIZE), map(map), ids(ids), 
          in_buffer_offsets(in_offsets_start, in_offsets_end), 
          in_buffer_ids(in_ids_start, in_ids_end), 
          m_buffer(nullptr), converted_buffer(nullptr), list_no(list_no) {}

};


struct AsyncReadRequests_Partial_PQDecode{
	std::vector<AsyncRequest_Partial> list_requests;
	std::vector<PageBuffer<uint8_t>> page_buffers;
	std::function<void()> pq_callback;
	std::function<void(AsyncRequest_Partial* requested, void* buffer)> cal_callback;

	std::function<void()> dp_callback;

	void fill_buffer(){
		if(list_requests.size() == 0)
			return;
			
		for(int i = 0; i < list_requests.size(); i++){
            PageBuffer<uint8_t> page_buffer;
            page_buffer.ReservePageBuffer(list_requests[i].m_readSize);
            page_buffers.emplace_back(std::move(page_buffer));
			list_requests[i].m_buffer = (char*)page_buffers[i].GetBuffer();
		}
	}
};


struct AsyncRequest_IndexInfo{
	size_t page_num;
	std::uint64_t m_offset;
    std::uint64_t m_readSize;
	char* m_buffer;
	size_t list_no;
	size_t list_size;

	AsyncRequest_IndexInfo() 
        : page_num(0), m_offset(0), m_readSize(0), m_buffer(nullptr), list_no(0), list_size(0) {}

    AsyncRequest_IndexInfo(size_t p_num, std::uint64_t offset, std::uint64_t readSize, 
                           char* buffer, size_t l_no, size_t l_size)
        : page_num(p_num), m_offset(offset), m_readSize(readSize), 
          m_buffer(buffer), list_no(l_no), list_size(l_size) {}
};

struct AsyncRequests_IndexInfo{
	std::vector<AsyncRequest_IndexInfo> info_requests;
	std::vector<PageBuffer<uint8_t>> page_buffers;
	void fill_buffer(){
		if(info_requests.size() == 0)
			return;
			
		for(int i = 0; i < info_requests.size(); i++){
            PageBuffer<uint8_t> page_buffer;
            page_buffer.ReservePageBuffer(info_requests[i].m_readSize); 
            page_buffers.emplace_back(std::move(page_buffer)); 
			info_requests[i].m_buffer = (char*)page_buffers[i].GetBuffer();
		}
	}

};


struct AsyncRequest_Uncached : AsyncRequest_Partial{
	char* info_buffer; 
	std::uint64_t m_offset;
    std::uint64_t m_readSize;
};


}
#endif
