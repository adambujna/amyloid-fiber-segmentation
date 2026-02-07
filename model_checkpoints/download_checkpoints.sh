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


BASE_URL="https://zenodo.org/records/16533347/files"

CHECKPOINTS=(
    "sam2.1tiny-fibersegmentation_sam2.1_t.pt"
    "sam2.1tiny-finetuned_sam2.1_t.pt"
    "yolo11large-fibersegmentation.pt"
    "yolo11large-finetuned.pt"
    "yolo11nano-fibersegmentation.pt"
    "yolo11nano-finetuned.pt"
)


echo "Downloading model checkpoints..."
for FILE in "${CHECKPOINTS[@]}"; do
    echo "Downloading $FILE..."
    CMD "$BASE_URL/$FILE?download=1" "$FILE" || { echo "Failed to download model from $FILE"; exit 1; }
done


echo "Successfully downloaded checkpoints."
