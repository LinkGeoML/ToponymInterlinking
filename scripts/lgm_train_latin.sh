#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Comparison-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_latin_EU_NA_100k.txt --ev lSimilarityMetrics --onlyLATIN --sort --canonical

echo ""
echo "Classification-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity_latin_EU_NA_100k.txt --ev customFEMLExtended --onlyLATIN --sort --canonical
