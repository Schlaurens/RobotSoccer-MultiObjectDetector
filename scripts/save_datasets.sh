#!/bin/bash

# Check if the parent directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

parent_dir="$1"

# Check if the parent directory exists
if [ ! -d "$parent_dir" ]; then
    echo "Error: Directory '$parent_dir' does not exist."
    exit 1
fi

# Loop through each subdirectory
for dir in "$parent_dir"/*/; do
    if [ -d "$dir" ]; then
        echo "Generating .tfrecords file for directory: $dir"
        start_time=$(date +%s)
        uv run src/save_dataset.py "$dir"
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Time taken: $elapsed seconds"
        echo "----------------------------------------"
    fi
done