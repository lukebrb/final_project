from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def logistic_regression():
    C_list = [10**x for x in range(-8, 5)] 
    gridsearch_params = [
    {
        'C': C_list,
        'penalty': ['l2'],
        'max_iter': [3000]
    }, 
    {
        'penalty': ['none'],
        'max_iter': [3000]
    },
    {
        'C': C_list,
        'penalty': ['l1'],
        'solver': ['liblinear'],
        'max_iter': [3000]
    }
    ]

    return (LogisticRegression(penalty='none'), gridsearch_params)

def SVM():
    C_list = [10**x for x in range(-7, 4)] 
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3,1e-2]

    gridsearch_params = [{
        'kernel': ['rbf', 'linear'],
        'C': C_list,
        'gamma': gamma_list,
    },
    {
        'kernel': ['poly'],
        'C': C_list,
        'gamma': gamma_list,
        'degree': np.linspace(1, 6, num=5)
    }
    ]

    return (SVC(kernel='rbf'), gridsearch_params)

def perceptron():
    gridsearch_params = [{
        'alpha': np.logspace(-14, 0, 5, base=2.0)
    }]

    return (Perceptron(), gridsearch_params)

def decision_tree():
    gridsearch_params = {
        'splitter': ['random', 'best'],
        'criterion': ['gini', 'entropy']
    }
    return (DecisionTreeClassifier(), gridsearch_params)



project_algorithms = [logistic_regression, SVM, perceptron, decision_tree]