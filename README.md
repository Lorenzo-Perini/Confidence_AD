# Confidence_AD

`Confidence_AD` (Confidence in Anomaly Detection) is a GitHub repository containing the **ExCeeD** [1] algorithm.
It refers to the paper titled *Quantifying the confidence of anomaly detectors in their example-wise predictions*. Read the pdf here: [[pdf](https://lirias.kuleuven.be/3059378?limo=0)].

## Abstract

Anomaly detection focuses on identifying examples in the data that somehow deviate from what is expected or typical. Algorithms for this task usually assign a score to each example that represents how anomalous the example is. Then, a threshold on the scores turns them into concrete predictions.
However, each algorithm uses a different approach to assign the scores, which makes them difficult to interpret and can quickly erode a user's trust in the predictions.
Here we introduce **ExCeeD** [1], an approach for assessing the reliability of any anomaly detector's example-wise predictions in two steps:
1) it transforms anomaly scores into outlier probabilities by using a Bayesian approach, and 
2) it derives a confidence score for each exemple-wise prediction, which captures the anomaly detector's uncertainty in that prediction.

## Contents and usage

The repository contains:
- ExCeeD.py, a function that allows to get the confidence values;
- Notebook.ipynb, a notebook showing how to use ExCeeD on an artificial dataset;
- evaluate_ExCeeD.py, a function used to get the experimental results on benchmark datasets;
- Experiments.ipynb, a notebook showing how to run the experiments.

To use ExCeeD, import the github repository or simply download the files. You can also find the benchmark datasets inside the folder Benchmark_Datasets or at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)].


## EXample-wise ConfidEncE of anomaly Detectors (ExCeeD)

Given a dataset with attributes **X**, an unsupervised anomaly detector assigns to each example an anomaly score, representing its degree of anomalousness. In order to compute the confidence scores, **ExCeeD** [1] takes into account the anomaly scores. For this reason, the algorithm works with *any anomaly detector*, assuming that the anomaly scores follow the rule that **the higher is the score, the more anomalous is the example**. It can also be used in a semi-supervised setting, where examples are partially labeled and even in a supervised setting, where all the examples are labeled.

Given a training dataset **X_train** (with labels *y* or not), and a test dataset **X_test**, the algorithm is applied as follows:

```python
from pyod.models.knn import KNN
from ExCeeD import *

# Estimate the contamination factor (which has to be given), for instance with
contamination = sum(y)/len(y)

# Train an anomaly detector (for instance, here we use kNNO)
detector = KNN().fit(X_train)

# Compute the anomaly scores in the training set
train_scores_knno = detector.decision_function(X_train)

# Compute the anomaly scores in the test set
test_scores_knno = detector.decision_function(X_test)

# Predict the class of each test example
prediction_knno = detector.predict(X_test)

# Estimate the confidence in class predictions with ExCeeD
knno_confidence = ExCeeD(train_scores_knno, test_scores_knno, prediction_knno, contamination)
```

## Dependencies

The `ExCeeD` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)


## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be)


## References

[1] Perini, L., Vercruyssen, V., Davis, J.: *Quantifying the confidence of anomaly detectors in their example-wise predictions.* In: The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. Springer Verlag (2020)
