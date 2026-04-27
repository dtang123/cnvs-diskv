// faiss/impl/S3IOReader.h
#pragma once

#include <faiss/impl/io.h>
#include <faiss/impl/S3IOStats.h>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/core/auth/AWSCredentialsProvider.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>

namespace faiss {

// ─────────────────────────────────────────────────────────────────────────────
// S3IOReader — drop-in replacement for FileIOReader that reads from S3/MinIO
// Usage:
//   S3IOReader reader("s3://warehouse/diskv/gist1m_0.index");
//   faiss::Index* index = faiss::read_index(&reader);
// ─────────────────────────────────────────────────────────────────────────────
struct S3IOReader : faiss::IOReader {

    std::shared_ptr<Aws::S3::S3Client> s3_client_;
    std::string bucket_;
    std::string key_;
    uint64_t    file_size_ = 0;
    uint64_t    pos_       = 0;

    // Full content buffer — index metadata files are small (few MB)
    // so we prefetch the entire file once on open for fast sequential reads
    std::vector<char> buffer_;
    bool              buffered_ = false;

    explicit S3IOReader(const std::string& s3_url) {
        // Parse s3://bucket/key
        const std::string prefix = "s3://";
        if (s3_url.rfind(prefix, 0) != 0)
            throw std::runtime_error("S3IOReader: URL must start with s3://");
        std::string rest = s3_url.substr(prefix.size());
        auto slash = rest.find('/');
        if (slash == std::string::npos)
            throw std::runtime_error("S3IOReader: no key in URL: " + s3_url);
        bucket_ = rest.substr(0, slash);
        key_    = rest.substr(slash + 1);

        // filename field used by FAISS internally for error messages

        // Build S3 client
        const char* endpoint = std::getenv("MINIO_ENDPOINT");
        const char* region   = std::getenv("AWS_DEFAULT_REGION");
        const char* key_id   = std::getenv("AWS_ACCESS_KEY_ID");
        const char* secret   = std::getenv("AWS_SECRET_ACCESS_KEY");

        Aws::Client::ClientConfiguration cfg;
        cfg.region = region ? region : "us-east-2";
        cfg.maxConnections = 4;

        bool use_minio = (endpoint != nullptr);
        if (use_minio) {
            std::string ep(endpoint);
            bool is_http = (ep.rfind("http://", 0) == 0);
            if (ep.rfind("http://",  0) == 0) ep = ep.substr(7);
            if (ep.rfind("https://", 0) == 0) ep = ep.substr(8);
            cfg.endpointOverride = ep;
            cfg.scheme = is_http ? Aws::Http::Scheme::HTTP
                                 : Aws::Http::Scheme::HTTPS;
        }

        if (key_id && secret) {
            Aws::Auth::AWSCredentials creds(key_id, secret);
            s3_client_ = std::make_shared<Aws::S3::S3Client>(
                creds, cfg,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                !use_minio);
        } else {
            s3_client_ = std::make_shared<Aws::S3::S3Client>(
                cfg,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                !use_minio);
        }

        // HEAD to get file size
        Aws::S3::Model::HeadObjectRequest head_req;
        head_req.SetBucket(bucket_.c_str());
        head_req.SetKey(key_.c_str());
        auto outcome = s3_client_->HeadObject(head_req);
        if (!outcome.IsSuccess()) {
            throw std::runtime_error(
                "S3IOReader: cannot HEAD s3://" + bucket_ + "/" + key_ +
                " : " + outcome.GetError().GetMessage());
        }
        file_size_ = static_cast<uint64_t>(
            outcome.GetResultWithOwnership().GetContentLength());

        std::cout << "[S3IOReader] Opened s3://" << bucket_ << "/" << key_
                  << " size=" << file_size_ << " bytes\n";

        // Prefetch entire file into buffer — index metadata is small (MB range)
        // This makes sequential FAISS deserialization fast (no per-read latency)
        prefetch_all();
    }

    void prefetch_all() {
        if (file_size_ == 0) { buffered_ = true; return; }

        Aws::S3::Model::GetObjectRequest req;
        req.SetBucket(bucket_.c_str());
        req.SetKey(key_.c_str());
        // No range header = fetch entire object

        auto outcome = s3_client_->GetObject(req);
        if (!outcome.IsSuccess()) {
            throw std::runtime_error(
                "S3IOReader: GET failed for s3://" + bucket_ + "/" + key_ +
                " : " + outcome.GetError().GetMessage());
        }

        buffer_.resize(file_size_);
        auto result = outcome.GetResultWithOwnership();
        auto& body  = result.GetBody();
        body.read(buffer_.data(), static_cast<std::streamsize>(file_size_));
        uint64_t got = static_cast<uint64_t>(body.gcount());

        if (got != file_size_) {
            std::cerr << "[S3IOReader] WARNING: expected " << file_size_
                      << " bytes but got " << got << "\n";
            buffer_.resize(got);
            file_size_ = got;
        }

        buffered_ = true;
        std::cout << "[S3IOReader] Prefetched " << got << " bytes\n";
    }

    // ── IOReader interface ────────────────────────────────────────────────────
    size_t operator()(void* ptr, size_t size, size_t nitems) override {
        size_t total = size * nitems;
        if (total == 0) return 0;

        if (!buffered_)
            throw std::runtime_error("S3IOReader: buffer not ready");

        // Clamp to available data
        uint64_t available = (pos_ < file_size_) ? (file_size_ - pos_) : 0;
        uint64_t to_copy   = std::min((uint64_t)total, available);

        if (to_copy > 0) {
            std::memcpy(ptr, buffer_.data() + pos_, to_copy);
            pos_ += to_copy;
        }

        return to_copy / size;  // return number of complete items read
    }

    int filedescriptor() override {
        return -1;  // no file descriptor for S3
    }
};

} // namespace faiss
