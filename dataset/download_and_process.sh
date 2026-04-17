#!/bin/bash

# Base directories
DATASET_DIR="/home/longnhat/ChessAI/dataset"
DATA_2700_DIR="$DATASET_DIR/data_2700"

mkdir -p "$DATA_2700_DIR"

# List of URLs specified by user
URLS=(
    "https://database.nikonoel.fr/lichess_elite_2021-08.zip"
    "https://database.nikonoel.fr/lichess_elite_2021-09.zip"
    "https://database.nikonoel.fr/lichess_elite_2021-10.zip"
    "https://database.nikonoel.fr/lichess_elite_2021-11.zip"
    "https://database.nikonoel.fr/lichess_elite_2021-12.zip"
    "https://database.nikonoel.fr/lichess_elite_2024-12.zip"
)

for URL in "${URLS[@]}"; do
    filename=$(basename "$URL")
    date_str=$(echo "$filename" | grep -oP '\d{4}-\d{2}')
    
    echo "========================================"
    echo "Processing $date_str..."
    
    # 1. Download
    aria2c -x 16 -s 16 -k 1M -d "$DATASET_DIR" "$URL"
    
    # 2. Extract
    echo "Extracting $filename..."
    # get the name of the file inside zip to safely move/delete it
    pgn_file=$(unzip -Z1 "$DATASET_DIR/$filename" | head -n 1)
    unzip -o "$DATASET_DIR/$filename" -d "$DATASET_DIR"
    
    # 3. Filter using awk
    filtered_out="$DATA_2700_DIR/filtered_lichess_elite_${date_str}.pgn"
    if [ -n "$pgn_file" ] && [ -f "$DATASET_DIR/$pgn_file" ]; then
        echo "Filtering $pgn_file to $filtered_out (requiring BOTH players > 2700)..."
        awk -f "$DATASET_DIR/filter_both_2700.awk" "$DATASET_DIR/$pgn_file" > "$filtered_out"
    else
        echo "Error: PGN file $pgn_file not found after extraction."
    fi
    
    # 4. Cleanup heavily memory-consuming raw files
    echo "Cleaning up raw zip and PGN files for $date_str..."
    rm -f "$DATASET_DIR/$filename"
    if [ -n "$pgn_file" ]; then
        rm -f "$DATASET_DIR/$pgn_file"
    fi
done

echo "========================================"
echo "All files systematically downloaded, filtered, and cleaned up!"
echo "Triggering the HDF5 conversion pipeline for newly filtered datasets..."

source /home/longnhat/ChessAI/venv/bin/activate
python "$DATASET_DIR/prepare_training_data.py"

echo "PIPELINE COMPLETED SUCCESSFULLY!"
