#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Comparison-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_original_100k.csv -e all

echo ""
echo "Classification-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_original_100k.csv --ev customFEML -e all
