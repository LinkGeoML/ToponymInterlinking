#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Comparison-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity.txt --onlyLATIN

echo ""
echo "Classification-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity.txt --ev customFEML --onlyLATIN --ml lsvm,dt,nn,nb

echo ""
echo "Classification-based interlinking...Random Forests"
python ../feml.py -d datasets/dataset-string-similarity.txt --onlyLATIN --ev customFEML --ml rf 

echo ""
echo "Classification-based interlinking...Extremely Randomized Trees"
python ../feml.py -d datasets/dataset-string-similarity.txt --onlyLATIN --ev customFEML --ml et

echo ""
echo "Classification-based interlinking...Gradient Boosted Trees"
python ../feml.py -d datasets/dataset-string-similarity.txt --onlyLATIN --ev customFEML --ml xgboost
