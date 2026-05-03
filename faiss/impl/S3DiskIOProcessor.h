#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/core/auth/AWSCredentialsProvider.h>

namespace faiss {

template<typename ValueType>
struct IVF_S3DiskIOSearchProcessor : DiskIOProcessor {

    std::shared_ptr<Aws::S3::S3Client> s3_client_;
    std::string s3_bucket_;
    std::string s3_key_;

    AsyncReadRequests_Partial_PQDecode* partial_diskRequests;
    AsyncReadRequests_Full_PQDecode*    full_diskRequests;
    AsyncRequests_IndexInfo*            info_diskRequests;
    
    faiss::InvertedLists* invlists = nullptr;

    void set_invlists(faiss::InvertedLists* il) {
        this->invlists = il;
    }

    IVF_S3DiskIOSearchProcessor(std::string disk_path, size_t d)
        : DiskIOProcessor(disk_path, d) {
        // Parse s3://bucket/key
        const std::string prefix = "s3://";
        if (disk_path.rfind(prefix, 0) != 0)
            throw std::runtime_error("IVF_S3DiskIOSearchProcessor: path must start with s3://");
        std::string rest = disk_path.substr(prefix.size());
        auto slash = rest.find('/');
        if (slash == std::string::npos)
            throw std::runtime_error("IVF_S3DiskIOSearchProcessor: no key in S3 URL");
        s3_bucket_ = rest.substr(0, slash);
        s3_key_    = rest.substr(slash + 1);
    }

    void initial(std::uint64_t /*maxIOSize*/    = (1 << 20),
                 std::uint32_t /*maxReadRetries*/ = 2,
                 std::uint32_t /*maxWriteRetries*/ = 2,
                 std::uint16_t /*threadPoolSize*/  = 4) override {

        const char* endpoint = std::getenv("MINIO_ENDPOINT");
        const char* region   = std::getenv("AWS_DEFAULT_REGION");
        const char* key_id   = std::getenv("AWS_ACCESS_KEY_ID");
        const char* secret   = std::getenv("AWS_SECRET_ACCESS_KEY");

        Aws::Client::ClientConfiguration cfg;
        cfg.maxConnections = 50;
	cfg.connectTimeoutMs = 3000;
	cfg.region = region ? region : "us-east-2";

        bool use_minio = (endpoint != nullptr);
        if (use_minio) {
            std::string ep(endpoint);
            bool is_http = (ep.rfind("http://", 0) == 0);
            if (ep.rfind("http://",  0) == 0) ep = ep.substr(7);
            if (ep.rfind("https://", 0) == 0) ep = ep.substr(8);
            cfg.endpointOverride = ep;
            cfg.scheme = is_http ? Aws::Http::Scheme::HTTP
                                 : Aws::Http::Scheme::HTTPS;
            cfg.maxConnections = 128;
        }

        if (key_id && secret) {
            Aws::Auth::AWSCredentials creds(key_id, secret);
            s3_client_ = std::make_shared<Aws::S3::S3Client>(
                creds, cfg,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                /*useVirtualAddressing=*/!use_minio);
        } else {
            s3_client_ = std::make_shared<Aws::S3::S3Client>(
                cfg,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                /*useVirtualAddressing=*/!use_minio);
        }
        std::cout << "[S3IO] bucket=" << s3_bucket_
                  << " key=" << s3_key_ << std::endl;
    }

    // ── Core range GET ────────────────────────────────────────────────────
    bool s3_range_read(uint64_t offset, uint64_t size, void* buf) const {
        std::string range = "bytes=" + std::to_string(offset) + "-"
                          + std::to_string(offset + size - 1);
        Aws::S3::Model::GetObjectRequest req;
        req.SetBucket("warehouse"); 
        req.SetKey("diskv/gist1m_0.clustered");
	//req.SetBucket(s3_bucket_.c_str());
        //req.SetKey(s3_key_.c_str());
        req.SetRange(range.c_str());

	// printf("[S3DiskIOProcessor] Expected size: %ld bytes\n", size);
        auto outcome = s3_client_->GetObject(req);
        if (!outcome.IsSuccess()) {
            std::cerr << "[S3IO] GET failed range=" << range
                      << " : " << outcome.GetError().GetMessage() << "\n";
            return false;
        }

	faiss::s3_io_stats.total_requests.fetch_add(1, std::memory_order_relaxed);
        faiss::s3_io_stats.total_bytes_read.fetch_add(size, std::memory_order_relaxed);

        auto result = outcome.GetResultWithOwnership();
        auto& body  = result.GetBody();
        body.read(static_cast<char*>(buf), static_cast<std::streamsize>(size));
        //printf("[DEBUG] First 4 bytes from S3: %02x %02x %02x %02x\n", 
           // ((uint8_t*)buf)[0], ((uint8_t*)buf)[1], ((uint8_t*)buf)[2], ((uint8_t*)buf)[3]);
	uint64_t got = static_cast<uint64_t>(body.gcount());
	if (got > size) {
            std::cerr << "[S3IO] OVERFLOW: requested=" << size 
                      << " got=" << got << " offset=" << offset << "\n";
            return false;
        }
        if (got != size) {
            std::cerr << "[S3IO] SHORT READ: requested=" << size 
                      << " got=" << got << " offset=" << offset << "\n";
            return false;
        }
	return (static_cast<uint64_t>(body.gcount()) == size);
    }

    void submit_partially() {
        int n = partial_diskRequests->list_requests.size();
        faiss::s3_io_stats.partial_requests.fetch_add(n, std::memory_order_relaxed);
    
        if (n == 0) {
            partial_diskRequests->pq_callback();  // ← only once
            return;
        }
    
        partial_diskRequests->fill_buffer();
    
        // ── Parallel S3 GETs ──────────────────────────────────────────────────────
        std::vector<std::future<bool>> futures;
        futures.reserve(n);
        for (int i = 0; i < n; i++) {
            AsyncRequest_Partial* req =
                partial_diskRequests->list_requests.data() + i;
            char*    buf    = req->m_buffer;
            uint64_t offset = req->m_offset;
            uint64_t size   = req->m_readSize;
            futures.push_back(std::async(std::launch::async,
                [this, offset, size, buf]() -> bool {
                    return s3_range_read(offset, size, buf);
                }));
        }
    
        // pq_callback overlaps with IO — called ONCE while futures are running
        partial_diskRequests->pq_callback();
    
        // Wait for all parallel GETs
        for (auto& f : futures) f.get();
    
        // cal_callbacks after all IO done
        for (int i = 0; i < n; i++) {
            AsyncRequest_Partial* req =
                partial_diskRequests->list_requests.data() + i;
            partial_diskRequests->cal_callback(req, req->m_buffer);
        }
        AsyncRequest_Partial* req = partial_diskRequests->list_requests.data() + 0; // check first one
        idx_t* id_ptr = (idx_t*)req->m_buffer;
        partial_diskRequests->dp_callback();  // ← once at end, no pq_callback here
    }
    
    void submit_fully(int num) {
        if (num <= 0) {
            full_diskRequests->dp_callback();
            return;
        }
    
        faiss::s3_io_stats.full_requests.fetch_add(num, std::memory_order_relaxed);
    
        int n = full_diskRequests->list_requests.size();
        full_diskRequests->page_buffers.clear();
        full_diskRequests->page_buffers.reserve(n);
    
        for (int i = 0; i < n; i++) {
            PageBuffer<uint8_t> pb;
            pb.ReservePageBuffer(full_diskRequests->list_requests[i].m_readSize);
            full_diskRequests->page_buffers.emplace_back(std::move(pb));
        }
        full_diskRequests->fill_buffer();
    
        // ── Parallel S3 GETs ──────────────────────────────────────────────────────
        std::vector<std::future<bool>> futures;
        futures.reserve(n);
        for (int i = 0; i < n; i++) {
            AsyncRequest_Full_Batch* req =
                full_diskRequests->list_requests.data() + i;
            void*    buf    = full_diskRequests->page_buffers[i].GetBuffer();
            uint64_t offset = req->m_offset;
            uint64_t size   = req->m_readSize;
            futures.push_back(std::async(std::launch::async,
                [this, offset, size, buf]() -> bool {
                    return s3_range_read(offset, size, buf);
                }));
        }
    
        // Wait for all parallel GETs
        for (int i = 0; i < num; i++) futures[i].get();
    
	// cal_callbacks after all IO done
        for (int i = 0; i < n; i++) {
            AsyncRequest_Full_Batch* req =
                full_diskRequests->list_requests.data() + i;
            AsyncRequest_Full* single = req->request_full.data();
            for (int j = 0; j < req->list_num; j++)
                full_diskRequests->cal_callback(single + j, single[j].m_buffer);
        }

	full_diskRequests->dp_callback();
        // NOTE: pq_callback NOT called here — search_o calls it before submit()
    }

    void submit_info() {
        int n = info_diskRequests->info_requests.size();
        std::vector<std::future<bool>> futures;
        futures.reserve(n);
        faiss::s3_io_stats.info_requests.fetch_add(n, std::memory_order_relaxed);

        for (int i = 0; i < n; i++) {
            AsyncRequest_IndexInfo* req =
                info_diskRequests->info_requests.data() + i;
            void* buf = info_diskRequests->page_buffers[i].GetBuffer();
            futures.push_back(std::async(std::launch::async,
                [this, req, buf]() -> bool {
                    return s3_range_read(req->m_offset, req->m_readSize, buf);
                }
            ));
        }
        for (auto& f : futures) f.get();
    }

    void submit(int num = -1) override {
        if      (num == -1) { submit_partially(); partial_diskRequests->page_buffers.clear(); }
        else if (num >= 0)  { submit_fully(num);  full_diskRequests->page_buffers.clear(); }
        else if (num == -2) { submit_info(); }
    }

    // ── Boilerplate — same as async version ──────────────────────────────
    float* convert_to_float_single(float* vector, void* disk_data, int begin) override {
        ValueType* src = reinterpret_cast<ValueType*>(disk_data) + begin;
        for (int i = 0; i < (int)d; i++)
            vector[i] = static_cast<float>(src[i]);
        return vector;
    }

    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        ValueType* src = reinterpret_cast<ValueType*>(disk_data);
        for (size_t i = 0; i < n * d; i++)
            vectors[i] = static_cast<float>(src[i]);
    }

    int process_page_transpage(int* vector_to_submit,
                               Page_to_Search* page_to_search,
                               size_t* vec_page_proj,
                               size_t len_p) override {
        int vector_size = sizeof(ValueType) * d;
        for (size_t i = 0; i < len_p; i++) {
            size_t off = vector_to_submit[i] * vector_size;
            page_to_search[i].first = off / PAGE_SIZE;
            page_to_search[i].last  = (off + vector_size - 1) / PAGE_SIZE;
            vec_page_proj[i] = page_to_search[i].first;
        }
        return len_p;
    }

    int get_per_page_element() override {
        return PAGE_SIZE / sizeof(ValueType);
    }

    void disk_io_partial_async_pq(
            AsyncReadRequests_Partial_PQDecode& r) override {
        r.fill_buffer();
        partial_diskRequests = &r;
    }

    void disk_io_full_async_pq(
            AsyncReadRequests_Full_PQDecode& r) override {
        full_diskRequests = &r;
    }

    void disk_io_info_async(AsyncRequests_IndexInfo& r) override {
        info_diskRequests = &r;
    }
};

}
