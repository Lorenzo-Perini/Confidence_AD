import numpy as np
import pandas as pd
from scipy.stats import binom

def ExCeeD(train_scores, test_scores, prediction, contamination):
    
    """
    Estimate the example-wise confidence according to the model ExCeeD provided in the paper.
    First, this method estimates the outlier probability through a Bayesian approach.
    Second, it computes the example-wise confidence by simulating to draw n other examples from the population.

    Parameters
    ----------
    train_scores   : list of shape (n_train,) containing the anomaly scores of the training set (by selected model).
    test_scores    : list of shape (n_test,) containing the anomaly scores of the test set (by selected model).
    prediction     : list of shape (n_test,) assuming 1 if the example has been classified as anomaly, 0 as normal.
    contamination  : float regarding the expected proportion of anomalies in the training set. It is the contamination factor.

    Returns
    ----------
    exWise_conf    : np.array of shape (n_test,) with the example-wise confidence for all the examples in the test set.
    
    """
    
    n = len(train_scores)
    n_anom = np.int(n*contamination) #expected anomalies
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)

    prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
    posterior_prob = prob_func(n_instances) #Outlier probability according to ExCeeD
    
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(posterior_prob)
    np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]) # if the example is classified as normal,
                                                                             # use 1 - confidence.
    
    return exWise_conf
