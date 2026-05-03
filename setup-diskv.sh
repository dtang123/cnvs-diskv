#!/bin/bash
# setup_diskv_minio.sh
# Source this file: source ~/setup_diskv_minio.sh

# ── MinIO / S3 credentials ────────────────────────────────────────────────────
export MINIO_ENDPOINT=http://127.0.0.1:19000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=password
export AWS_DEFAULT_REGION=us-east-2

# ── Configuration File Path ───────────────────────────────────────────────────
# This is the file the binary will read for search_estimate_factor, etc.
export DISKV_CONFIG_FILE=~/dataset_gist1m.sh

# ── Index paths (Optional if paths are inside the config file) ────────────────
export DISKV_INDEX_FILE=/dev/shm/diskv_metadata/gist1m_0.index
export DISKV_CLUSTERED_PATH=s3://warehouse/diskv/gist1m_0.clustered
export DISKV_CENTROID_PATH=/dev/shm/diskv_metadata/gist1m_centroid_0

# ── Query / ground truth ──────────────────────────────────────────────────────
export DISKV_QUERY_FILE=/path/to/gist_query.bin
export DISKV_GT_FILE=/path/to/gist_groundtruth.bin

# ── Search binary ─────────────────────────────────────────────────────────────
export DISKV_BIN=~/DiskV_VLDB26/build/demos/run_diskv_search

# ── Runtime linker path for AWS SDK ──────────────────────────────────────────
export LD_LIBRARY_PATH=/path/to/awssdk/lib:$LD_LIBRARY_PATH

# ── RAM disk setup ────────────────────────────────────────────────────────────
echo "Setting up RAM disk metadata..."
mkdir -p /dev/shm/diskv_metadata
if [[ ! -f /dev/shm/diskv_metadata/gist1m_0.index ]]; then
    cp ~/DiskV_index/gist1m_0.index    /dev/shm/diskv_metadata/
    echo "  Copied gist1m_0.index"
else
    echo "  gist1m_0.index already in RAM disk"
fi
if [[ ! -f /dev/shm/diskv_metadata/gist1m_centroid_0 ]]; then
    cp ~/DiskV_index/gist1m_centroid_0 /dev/shm/diskv_metadata/
    echo "  Copied gist1m_centroid_0"
else
    echo "  gist1m_centroid_0 already in RAM disk"
fi

# ── Verify MinIO ──────────────────────────────────────────────────────────────
echo "Checking MinIO..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    ${MINIO_ENDPOINT}/minio/health/live || echo "000")
if [[ "${HTTP_CODE}" != "200" ]]; then
    echo "  WARNING: MinIO not reachable (HTTP ${HTTP_CODE})"
    echo "  Run: cd ~/my-lab && docker compose up -d"
else
    echo "  MinIO OK (HTTP ${HTTP_CODE})"
fi

# ── Verify index in MinIO ─────────────────────────────────────────────────────
if ! ~/mc stat minio/warehouse/diskv/gist1m_0.clustered > /dev/null 2>&1; then
    echo "  WARNING: gist1m_0.clustered not found in MinIO"
    echo "  Run: ~/mc cp ~/DiskV_index/gist1m_0.clustered minio/warehouse/diskv/gist1m_0.clustered"
else
    echo "  Index file found in MinIO"
fi

# ── Print run command ─────────────────────────────────────────────────────────
echo ""
echo "Environment ready. Run the search using the config file:"
echo ""
echo "  \${DISKV_BIN} \${DISKV_CONFIG_FILE}"
echo ""
echo "Note: Ensure \${DISKV_CONFIG_FILE} has correct paths for S3 and metadata."
