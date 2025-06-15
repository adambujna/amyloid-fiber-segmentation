#!/bin/bash


if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the datasets."
    exit 1
fi

BASE_URL="https://liacs.leidenuniv.nl/~s3610233/thesis/"
ZIP_FILE_YOLO="$BASE_URL/yolo_dataset.zip"
ZIP_FILE_SAM="$BASE_URL/sam_dataset.zip"


echo "Downloading YOLO dataset..."
$CMD $ZIP_FILE_YOLO || { echo "Failed to download dataset from $ZIP_FILE_YOLO"; exit 1; }

echo "Downloading SAM dataset..."
$CMD $ZIP_FILE_SAM || { echo "Failed to download dataset from $ZIP_FILE_SAM"; exit 1; }

echo "Successfully downloaded datasets."
