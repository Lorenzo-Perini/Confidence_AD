import numpy as np
import pandas as pd
import random
import math
from scipy.stats import binom
from sklearn.model_selection import StratifiedKFold
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
from beta_calibration import*

def squash_proba(x, gamma):
    return 2**(-(x/gamma)**2)

def train_model(X_train, contamination, model):
    
    """
    Train the model based on the user's choice (KNN, IForest, LOF, OCSVM).

    Parameters
    ----------
    X_train       : list of shape (n_train, n_features) containing the training set with only features.
    contamination : float representing the expected proportion of anomalies in the training set.
    model         : string, claiming the model to use for evaluating the confidence. It can be one of: KNN, IForest, OCSVM, LOF.

    Returns
    ----------
    clf           : obj with the model trained on the training set.
    
    """

    n = len(X_train)
    if model == 'KNN':
        clf = KNN(n_neighbors=max(np.int(n*contamination),1), contamination = contamination).fit(X_train)
    elif model == 'IForest':
        clf = IForest(contamination = contamination, random_state = 331).fit(X_train)
    elif model == 'LOF':
        clf = LOF(contamination = contamination).fit(X_train)
    elif model == 'OCSVM':
        clf = OCSVM(contamination = contamination).fit(X_train)
    return clf


def empirical_frequencies(iterations, X_train, X_test, contamination, model):
    
    """
    Empirically estimate frequencies of class prediction for each example in the test set. The method does n_iter = iterations.

    Parameters
    ----------
    iterations    : int representing the number of iterations to make when empirically assessing the class frequencies.
    X_train       : list of shape (n_train, n_features) containing the training set with only features.
    X_test        : list of shape (n_test, n_features) containing the test set with only features.
    contamination : float representing the expected proportion of anomalies in the training set.
    model         : string, claiming the model to use for evaluating the confidence. It can be one of: KNN, IForest, OCSVM, LOF.

    Returns
    ----------
    final_freq    : list of shape (n_test,) containing the empirical frequencies for each example in the test set.
    
    """
    
    n = len(X_train)
    empirical_freq = np.zeros(len(X_test), np.double)
    clf = train_model(X_train, contamination, model)
    pred = clf.predict(X_test)
    for iteration in range(iterations):
        sizetset = random.sample(range(np.int(0.2*n), n), 1)
        indexes_tset = random.sample(range(n), sizetset[0])
        subX_train = X_train[indexes_tset]
        clf2 = train_model(subX_train, contamination, model) 
        prediction = clf2.predict(X_test)
        empirical_freq += prediction
    empirical_freq = [empirical_freq[i]/iterations if pred[i] == 1 else 1-empirical_freq[i]/iterations\
                      for i in range(len(X_test))]
    final_freq = [round(min(1,empirical_freq[i]),4) for i in range(len(X_test))]
    return final_freq
    

def confidence_estimation(X_train, y_train, X_test, y_test, contamination, model):
    
    """
    Estimate the example-wise confidence of different methods, included the model ExCeeD provided in the paper.
    First, we compute the outlier probabilities for all the methods but Calibrations. Then, we estimate the
    example-wise confidence for ExCeeD, ExCeed_sp (with second prior), ExCeeD with outlier probability computed
    through the linear, squashing and unify methods, and calibrated probabilities through Logistic (logcal),
    Isotonic (isocal) and Beta (betacal) Calibrations.

    Parameters
    ----------
    X_train       : list of shape (n_train, n_features) containing the training set with only features.
    y_train       : list of shape (n_train,) containing the actual labels for the training set. It is needed for Calibration.
    X_test        : list of shape (n_test, n_features) containing the test set with only features.
    y_test        : list of shape (n_test,) containing the actual labels for the test set. It is needed for Calibration.
    contamination : float representing the expected proportion of anomalies in the training set.
    model         : string, claiming the model to use for evaluating the confidence. It can be one of: KNN, IForest, OCSVM, LOF.

    Returns
    ----------
    exceed_conf    : example-wise confidence using ExCeeD (outlier probability computed by Bayesian Learning with uniform prior) 
    exceed_conf_sp : example-wise confidence using ExCeeD (outlier probability computed by Bayesian Learning with other prior)
    squash_conf    : example-wise confidence using ExCeeD (outlier probability computed by squashing function)
    linear_conf    : example-wise confidence using ExCeeD (outlier probability computed by linear function)
    unify_conf     : example-wise confidence using ExCeeD (outlier probability computed by unify method)
    logcal_conf    : example-wise calibrated probability using Logistic Calibration
    isocal_conf    : example-wise calibrated probability using Isotonic Calibration
    betacal_conf   : example-wise calibrated probability using Beta Calibration
    prediction     : list of class predictions with shape (n_test,)
    
    """
    
    n = len(X_train)
    clf = train_model(X_train, contamination, model)
    col = clf.decision_function(X_train)
    prediction = clf.predict(X_test)
    test_scores = clf.decision_function(X_test)
    train_scores = clf.decision_function(X_train)
    
    n_anom = np.int(n*contamination)
    m = 10/contamination
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)

    prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
    exceed_posterior = prob_func(n_instances)
    
    adj_prob_func = np.vectorize(lambda x: (10+x)/(m+n)) 
    exceed_posterior_sp = adj_prob_func(n_instances)
    
    unify_proba_anom = [x[1] for x in clf.predict_proba(X_test, method='unify')]

    linear_proba_anom = [x[1] for x in clf.predict_proba(X_test, method='linear')]

    tmp_score = sorted(col, reverse = True)
    gamma = tmp_score[min(np.int(n*contamination)-1,0)]
    squashing_proba_anom = [1-squash_proba(x, gamma) for x in clf.decision_function(X_test)]
        
    mapinto01_train = [1-squash_proba(x, gamma) for x in clf.decision_function(X_train)]
    mapinto01_test = [1-squash_proba(x, gamma) for x in clf.decision_function(X_test)]

    lr = LogisticRegression(C=99999999999)
    lr.fit(np.asarray(mapinto01_train).reshape(-1, 1), y_train)
    logistic_calibration = lr.predict_proba(np.asarray(mapinto01_test).reshape(-1, 1))[:,1]

    iso = IsotonicRegression()
    iso.fit(mapinto01_train, y_train)
    isotonic_calibration = np.nan_to_num(iso.predict(mapinto01_test), nan = 1.0)

    bc = BetaCalibration(parameters="abm")
    bc.fit(np.asarray(mapinto01_train).reshape(-1, 1), y_train)
    beta_calibration = bc.predict(np.asarray(mapinto01_test).reshape(-1, 1))
    
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    
    exceed_conf = conf_func(exceed_posterior)
    np.place(exceed_conf, prediction == 0, 1 - exceed_conf[prediction == 0])
    
    exceed_conf_sp = conf_func(exceed_posterior_sp)
    np.place(exceed_conf_sp, prediction == 0, 1 - exceed_conf_sp[prediction == 0])
    
    squash_conf = conf_func(squashing_proba_anom)
    np.place(squash_conf, prediction == 0, 1 - squash_conf[prediction == 0])    

    linear_conf = conf_func(linear_proba_anom)
    np.place(linear_conf, prediction == 0, 1 - linear_conf[prediction == 0])    

    unify_conf = conf_func(unify_proba_anom)
    np.place(unify_conf, prediction == 0, 1 - unify_conf[prediction == 0])    

    logcal_conf = np.asarray([round(x,4) if prediction[i] == 1 else round(1-x,4) for i,x in enumerate(logistic_calibration)])
    isocal_conf = np.asarray([round(x,4) if prediction[i] == 1 else round(1-x,4) for i,x in enumerate(isotonic_calibration)])
    betacal_conf = np.asarray([round(x,4) if prediction[i] == 1 else round(1-x,4) for i,x in enumerate(beta_calibration)])

    return exceed_conf, exceed_conf_sp, squash_conf, linear_conf, unify_conf, logcal_conf, isocal_conf, betacal_conf, prediction
    
    
def l2_distance(confidence, frequence):
    
    return sum((confidence - frequence)**2)/len(confidence)
    
    
def compute_confidence_error(model, Dtrain, y, splits = 5, iterations = 1000):
    
    """
    Evaluate the example-wise confidence of different methods, included the model ExCeeD provided in the paper.
    First, we split the data set in training and test set using a k fold crossvalidation. Second, we estimate the
    example-wise confidence for ExCeeD, ExCeed_sp (with second prior), ExCeeD with outlier probability computed
    through the linear, squashing and unify methods, and calibrated probabilities through Logistic (logcal),
    Isotonic (isocal) and Beta (betacal) Calibrations. Third, we empirically estimate frequencies of class prediction
    for each example. Finally, we evaluate the different methods by taking the mean L2 error per class.

    Parameters
    ----------
    model      : string, claiming the model to use for evaluating the confidence. It can be one of: KNN, IForest, OCSVM, LOF.
    Dtrain     : list of shape (n_samples, n_features) containing the dataset with only features.
    y          : list of shape (n_samples,) assuming 1 if the example is an anomaly, 0 if it is normal.
    splits     : int regarding the number of splits to make during the stratified crossvalidation phase.
    iterations : int representing the number of iterations to make when empirically assessing the class frequencies.

    Returns
    ----------
    l2_error   : dictionary with values [a,b], where a stand for the mean L2 error for normal examples, b for anomalies.
    
    """
    
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    
    l2_error = {'exceed': np.zeros(2, np.float), 'exceed_sp': np.zeros(2, np.float), 'squash': np.zeros(2, np.float),
                'linear': np.zeros(2, np.float), 'unify': np.zeros(2, np.float), 'logcal': np.zeros(2, np.float), 
                'isocal': np.zeros(2, np.float), 'betacal': np.zeros(2, np.float), 'baseline': np.zeros(2, np.float)}
    
    for train_index, test_index in skf.split(Dtrain, y):
        
        X_train, X_test = Dtrain[train_index], Dtrain[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        contamination = sum(y_train)/len(y_train) #the real contamination factor. We assume it is known.
        exceed_conf, exceed_conf_otherprior, squash_conf, linear_conf, unify_conf, logcal_conf, isocal_conf, betacal_conf,\
        prediction = confidence_estimation(X_train, y_train, X_test, y_test, contamination, model)
        frequence = np.asarray(empirical_frequencies(iterations, X_train, X_test, contamination, model))
        
        anomalies = np.where(prediction == 1)[0] #indexes of normals
        normals = np.where(prediction == 0)[0]   #indexes of anomalies
        
        l2_error['exceed'] += [l2_distance(exceed_conf[normals], frequence[normals])/splits,
                              l2_distance(exceed_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['exceed_sp'] += [l2_distance(exceed_conf_otherprior[normals],frequence[normals])/splits,
                            l2_distance(exceed_conf_otherprior[anomalies], frequence[anomalies])/splits]
       
        l2_error['squash'] += [l2_distance(squash_conf[normals], frequence[normals])/splits,
                          l2_distance(squash_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['linear'] += [l2_distance(linear_conf[normals], frequence[normals])/splits,
                          l2_distance(linear_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['unify'] += [l2_distance(unify_conf[normals], frequence[normals])/splits,
                          l2_distance(unify_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['logcal'] += [l2_distance(logcal_conf[normals], frequence[normals])/splits,
                          l2_distance(logcal_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['isocal'] += [l2_distance(isocal_conf[normals], frequence[normals])/splits,
                          l2_distance(isocal_conf[anomalies], frequence[anomalies])/splits]
        
        l2_error['betacal'] += [l2_distance(betacal_conf[normals], frequence[normals])/splits,
                          l2_distance(betacal_conf[anomalies], frequence[anomalies])/splits]

        l2_error['baseline'] += [l2_distance(np.ones(len(normals)), frequence[normals])/splits,
                          l2_distance(np.ones(len(anomalies)), frequence[anomalies])/splits]
        
    return l2_error
    