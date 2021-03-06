#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Comparison-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_original_100k.csv --ev lSimilarityMetrics -e all --sort --canonical

echo ""
echo "Classification-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_original_100k.csv --ev customFEMLExtended -e all --sort --canonical
