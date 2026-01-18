#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


if command -v wget &> /dev/null; then
    CMD() { wget "$1" -O "$2"; }
elif command -v curl &> /dev/null; then
    CMD() { curl -L "$1" -o "$2"; }
else
    echo "Please install wget or curl to download the datasets."
    exit 1
fi

BASE_URL="https://zenodo.org/records/16533418/files"
ZIP_FILE_YOLO="$BASE_URL/yolo_dataset.zip?download=1"
ZIP_FILE_SAM="$BASE_URL/sam_dataset.zip?download=1"
ZIP_FILE_REAL="$BASE_URL/real_dataset.zip?download=1"


echo "Downloading YOLO dataset..."
CMD "$ZIP_FILE_YOLO" "yolo_dataset.zip" || { echo "Failed to download dataset from $ZIP_FILE_YOLO"; exit 1; }

echo "Downloading SAM dataset..."
CMD "$ZIP_FILE_SAM" "sam_dataset.zip" || { echo "Failed to download dataset from $ZIP_FILE_SAM"; exit 1; }

echo "Downloading REAL dataset..."
CMD "$ZIP_FILE_REAL" "real_dataset.zip" || { echo "Failed to download dataset from $ZIP_FILE_REAL"; exit 1; }

echo "Successfully downloaded datasets."
