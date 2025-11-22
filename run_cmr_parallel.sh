#!/bin/bash

# This script runs cmr_grid_temp.py on all studies in parallel using GNU parallel
# Usage: ./run_cmr_parallel.sh [num_jobs]
# Example: ./run_cmr_parallel.sh 100

EXTRACTED_DIR="/home/masadi/projects_link/cmr_extracted"
SCRIPT="/home/masadi/cmr_grid_temp.py"
NUM_JOBS=${1:-100}  # Default to 100 jobs if not specified

# Create output directory if it doesn't exist
mkdir -p /home/masadi/projects_link/cmr_grids

# Find all study directories and run in parallel
find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | \
    parallel --jobs $NUM_JOBS --progress --eta \
    "python $SCRIPT {}"

echo "All studies processed!"

