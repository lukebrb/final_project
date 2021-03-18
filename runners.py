from multiprocessing import Pool
import time
from datetime import datetime
from typing import Any
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools

from classifiers import project_algorithms
from data_ingestion import ingestion_functions
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics
from dataclasses import dataclass
import joblib
import os


@dataclass
class Result:
    algo_name: str
    dataset_name: str
    trial_num: int
    X_train: Any
    Y_train: Any
    Y_train_pred: Any
    X_test: Any
    Y_test: Any
    Y_test_pred: Any
    # Runtime, in seconds 
    runtime: float
    """A pandas DataFrame that can be imported 
    """
    cv_results_: Any


def run_trial(with_params) -> Result:
    """Runs a given trial using a given algorithm. Fetches data from data_fetcher.

    Args:
        data_fetcher (fn -> tuple): [description]
        algorithm ([fn]): [description]
        num_trial ([type]): [description]

    Returns:
        [type]: [description]
    """
    start = datetime.now()

    data_fetcher, algorithm, num_trial = with_params

    (algo, params) = algorithm()
    X_train, X_test, Y_train, Y_test = data_fetcher()

    # GridSearchCV automatically does 5 kfold splits.
    search_results = GridSearchCV(algo, params)
    search_results.fit(X_train, Y_train)

    opt_classifier = search_results.best_estimator_
    opt_classifier.fit(X_train, Y_train)

    Y_train_pred = opt_classifier.predict(X_train)
    Y_test_pred = opt_classifier.predict(X_test)
    # Get metrics for the classifiers

    end = datetime.now()

    runtime = (end - start).total_seconds()
    return Result(
        algo_name=algorithm.__name__,
        dataset_name=data_fetcher.__name__,
        trial_num=num_trial,
        X_train = X_train,
        Y_train = Y_train,
        Y_train_pred = Y_train_pred,
        X_test=X_test,
        Y_test = Y_test,
        Y_test_pred = Y_test_pred,
        runtime=runtime ,
        cv_results_ = search_results.cv_results_
    )


def run_all_trials():
    trial_combinations = list(
        itertools.product(ingestion_functions, project_algorithms, list(range(5)))
    )
    # Runs all concurrently on different CPUs
    # My M1 Macbook Air has 8 cores, so 8 + 4 = 12
    YOUR_CPU_CORES = 8
    results = process_map(run_trial, trial_combinations, max_workers=YOUR_CPU_CORES + 4)

    # Single-threaded for easier debugging
    #results = [run_trial(tc) for tc in trial_combinations]

    timestamp = int(time.time())

    for result in tqdm(results, desc="Saving classifiers to disk..."):
        # Save the classifier to disk for use in a Jupyter Notebook
        folder_path = f"./classifier_cache/{timestamp}/{result.algo_name}/{result.dataset_name}"
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass

        result_filename = folder_path + f"/{result.trial_num}_cls.joblib.pkl"
        _ = joblib.dump(result, result_filename, compress=9)

if __name__ == "__main__":
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    run_all_trials()

