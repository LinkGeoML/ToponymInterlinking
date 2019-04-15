# ToponymInterlinking
This is the Python code used to implement and evaluate the proposed LinkGeoML models for Toponym Interlinking paper sumbitted in [SSTD2019](http://sstd2019.org/) (under review). The dataset folder contains the train datasets used for evaluation. For the test dataset, we used the one from the Toponym-Matching work (see [Setup](./README.md#setup)).
The **scripts** folder contains the evaluation setting used to execute the experiments and collect the results presented in the paper.

The source code was tested using Python 2.7 and Scikit-Learn 0.20.3 on a Linux server.

Setup
------------

Download the latest version from the [GitHub repository](https://github.com/LinkGeoML/ToponymInterlinking.git), change to the main directory and run:

```bash
    pip install -r requirements.txt
```

It should install all the required libraries automatically (*scikit-learn, numpy, pandas etc.*).

Change to the **datasets** folder, download the test dataset and unzip it:
```bash
    wget https://github.com/ruipds/Toponym-Matching/raw/master/dataset/dataset.zip
    wget https://github.com/ruipds/Toponym-Matching/raw/master/dataset/dataset.z01

    zip -FF dataset.zip  --out dataset.zip.fixed
    unzip dataset.zip.fixed
```

## Acknowledgements
The *datasetcreator.py* file, which is used to generate the train/test datasets and to compute the string similarity measures, is a slightly modified version of the one used in [Toponym-Matching](https://github.com/ruipds/Toponym-Matching) work and is under the MIT license.

## License
ToponymInterlinking is available under the MIT License (see LICENSE.txt).  

