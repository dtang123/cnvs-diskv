# Step 1: Make executable
I assume you have awssdk already installed. Link awssdk path using cmake
```
mkdir -p build
cd build
cmake -DAWSSDK_ROOT=/path/to/awssdk ..
make demo_script
make run_diskv_search
```
# Step 2: Make a .sh file for dataset
This file determines how the index is built and searched in later steps. I have provided a sample file: dataset\_gist1m.sh that is a configuration for the GIST1M dataset
# Step 3: Build index
This builds based on the configuration in the gist1m.sh file
```
./build/demos/demo_script 0 ./dataset_gist1m.sh
```
Refer to [Original DiskV Repository](https://github.com/YunanZzz/DiskV_VLDB26) for additional details about the index
# Step 4: Minio Setup
I assume you have installed mc. 
I have provided the docker-compose.yml file to create the Minio container in root directory. This will mimic an S3 bucket, allowing us to request from it using aws\_sdk. Use below command to start up bucket.
```
docker compose up -d
```
# Step 5: Move index to Minio
Copy the index to Minio
```
~/mc cp /path/to/gist1m_0.clustered minio/warehouse/diskv
```
# Step 6: Setup traffic control to mimic cloud native indexing
This will add latency to the requests to MinIO. This will mimic cloud native indexing
```
sudo tc qdisc add dev lo root netem delay 31ms 20ms distribution normal
```
# Step 7:  Setup Script
Copies metadata files to ramdisk and check minio is working. I have given this file ask setup-diskv.sh 
```
source ./setup-diskv.sh
```
# Step 8: Run
```
\${DISKV_BIN} \${DISKV_CONFIG_FILE}
```
