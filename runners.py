from multiprocessing import Pool
from typing import Any
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools

from classifiers import project_algorithms
from data_ingestion import ingestion_functions
from sklearn.model_selection import KFold, GridSearchCV
from dataclasses import dataclass


@dataclass
class Result:
    algo_name: str
    dataset_name: str
    trial_num: int
    optimal_classifier: Any


def run_trial(with_params):
    """Runs a given trial using a given algorithm. Fetches data from data_fetcher.

    Args:
        data_fetcher (fn -> tuple): [description]
        algorithm ([fn]): [description]
        num_trial ([type]): [description]

    Returns:
        [type]: [description]
    """

    data_fetcher, algorithm, num_trial = with_params
    print(f"Running {algorithm.__name__} on {data_fetcher.__name__}: Trial {num_trial}")

    (algo, params) = algorithm()
    X_train, X_test, Y_train, Y_test = data_fetcher()

    # GridSearchCV automatically does 5 kfold splits.
    search_results = GridSearchCV(algo, params)
    search_results.fit(X_train, Y_train)

    opt_classifier = search_results.best_estimator_
    opt_classifier.fit(X_train, Y_train)

    return Result(
        algo_name=algorithm.__name__,
        dataset_name=data_fetcher.__name__,
        trial_num=num_trial,
        optimal_classifier=opt_classifier,
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
    #for tc in trial_combinations:
    #    run_trial(tc)


if __name__ == "__main__":
    run_all_trials()

