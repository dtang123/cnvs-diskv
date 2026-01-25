#!/bin/bash

config_file="./demos/dataset.sh"
program_exec="./build/demos/demo_script"   # e.g. demo_script

# Check whether recommendation is needed first
use_rec=$(grep '^use_recommend=' "$config_file" | cut -d= -f2)
if [[ "$use_rec" == "true" ]]; then
    echo "Detected use_recommend=true, running disk speed test..."
    bash ./demos/disk_speed.sh "$config_file"
fi

# Then launch the main program
# Assume the first argument is build/search and the second is the config path
if [[ "$1" == "build" ]]; then
    $program_exec 0 "$config_file"
elif [[ "$1" == "search" ]]; then
    $program_exec 1 "$config_file"
else
    echo "Usage: $0 [build|search]"
    exit 1
fi
