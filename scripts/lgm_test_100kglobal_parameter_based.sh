#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Comparison-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity.txt -e all --sort --canonical

echo ""
echo "Classification-based interlinking..."
python ../feml.py -d datasets/dataset-string-similarity.txt --ev customFEMLExtended -e all --sort --canonicala --ml lsvm,dt,nn,nb

echo ""
echo "Classification-based interlinking...Random Forests"
python ../feml.py -d dataset-string-similarity.txt -e all --ev customFEMLExtended --ml rf --sort --canonical -f False,False,False,False,False,True,False,False,True,False,False,True,True,True,False,True,False,False,False,False,False,False,True,False,True,True,False,True,False,False,False,False,False,False,False,False,True,False,False,False

echo ""
echo "Classification-based interlinking...Extremely Randomized Trees"
python ../feml.py -d dataset-string-similarity.txt -e all --ev customFEMLExtended --ml et --sort --canonical -f False,False,False,False,False,False,False,False,True,True,False,True,True,True,False,True,False,False,False,False,False,True,True,True,True,True,False,True,False,False,False,False,False,False,False,False,True,False,False,False

echo ""
echo "Classification-based interlinking...Gradient Boosted Trees"
python ../feml.py -d dataset-string-similarity.txt -e all --ev customFEMLExtended --ml xgboost --sort --canonical -f True,False,False,False,False,True,True,False,False,True,True,False,True,True,False,False,False,True,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,True,True,False,False
