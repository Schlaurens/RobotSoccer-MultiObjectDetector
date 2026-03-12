#!/bin/bash
# Suppress TensorFlow warnings and messages
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

# Check if the parent directory is provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <parent_directory> <dest_directory> <im_height> <im_width> <cell_height> <cell_width>"
    exit 1
fi

parent_dir="$1"
dest_dir="$2"
im_height="$3"
im_width="$4"
cell_height="$5"
cell_width="$6"


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
        uv run src/dataset/save_dataset.py --src_dir "$dir" --dest_dir "$dest_dir" --image_res $im_height $im_width --cell_dims $cell_height $cell_width
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Time taken: $elapsed seconds"
        echo "----------------------------------------"
    fi
done