# Celebrity Profiling

>The project is from TDT4310 Intelligent Text Analytics and Language Understanding at NTNU. The system compares the results of Perceptron, Stocatich Desending Gradient, Passive Aggressive Classifier, Mutlinomial Naive Bayes and Bernoulli Naive Bayes with the data of multiple celebrity users' twitter feed
.
PS: The code is written in Python 3.8.5 and will therefore need at least Python3 and upward to function. Additionally, the codebase has a part where file manipulation is performed and could need some form of permissions to operate. The system was created in a Linux( Ubuntu 20.04.2 LTS) environment and has not been tested on Windows nor Mac systems.



## Dataset

The data usedfor the project is from [PAN20: Celebrity Profiling 2020](https://pan.webis.de/clef20/pan20-web/celebrity-profiling.html). 
Access is requested and granted by [Zenodo](https://zenodo.org/record/4461887).

## Prerequisites

- Numpy
- Pandas
- Sklearn
- Ujson
- Additional libraries +

Packages can be installed using the requirement.txt file with the command
```python
pip install -r ./requirements.txt
```

## Usage

`To use, run the command:
```python
python main.py
```
Changes to the test/training split can be done in the configuration file called "config.py". Changes on other setting have not been performed and are unknown if it will be stable. As of now, it does every prosses in the pipeline.

To change the split ratio, the files inside data/pickled_data/test and data/pickled_data/train must be removed(ends with .pkl).

## Acknowledgements 
* PAN20 Authorship Analysis: Celebrity Profiling for the task
* [Zenodo](https://zenodo.org/record/4461887) for accsess to the data