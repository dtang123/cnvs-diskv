#!/bin/bash

# Read test_speed_file path from config
conf_file="$1"
search_index_store_path=$(grep '^test_speed_file=' "$conf_file" | cut -d= -f2)

read_size_mb=2048
if [ ! -f "$search_index_store_path" ]; then
    echo "Error: $search_index_store_path does not exist!"
    exit 1
fi

echo "Testing sequential read speed for the first $read_size_mb MB of $search_index_store_path..."

dd if="$search_index_store_path" of=/dev/null bs=1M count=$read_size_mb iflag=direct 2>&1 | tee dd_read.log

# Extract read speed in MB/s
read_speed=$(grep -i 'copied' dd_read.log | awk '{print $(NF-1)}')
unit=$(grep -i 'copied' dd_read.log | awk '{print $NF}')


if [[ $unit == "GB/s" ]]; then
    read_speed=$(echo "$read_speed * 1024" | bc)
fi

# Recommendation: SSD > 600 MB/s, NVMe > 1800 MB/s
top_value=80
if (( $(echo "$read_speed > 1000" | bc -l) )); then
    top_value=1     # NVMe
elif (( $(echo "$read_speed > 300" | bc -l) )); then
    top_value=1     # SSD
else
    top_value=80    # HDD
fi

echo "Disk read speed is $read_speed MB/s, recommended search_top=$top_value"

# Automatically update the config file (replace search_top field)
if grep -q '^search_top=' "$conf_file"; then
    sed -i "s/^search_top=.*/search_top=$top_value/" "$conf_file"
else
    echo "search_top=$top_value" >> "$conf_file"
fi
