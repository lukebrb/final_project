from multiprocessing import Pool
from typing import Any
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools

from classifiers import project_algorithms
from data_ingestion import ingestion_functions
from sklearn.model_selection import KFold, GridSearchCV
from dataclasses import dataclass
import joblib
import os


@dataclass
class Result:
    algo_name: str
    dataset_name: str
    trial_num: int
    optimal_classifier: Any
    test_data: tuple[Any, Any]


def run_trial(with_params) -> Result:
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
        test_data=(X_test, Y_test)
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

    for result in tqdm(results, desc="Saving classifiers to disk..."):
        # Save the classifier to disk for use in a Jupyter Notebook
        folder_path = f"./classifier_cache/{result.algo_name}/{result.dataset_name}"
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            pass

        cls_filename = folder_path + f"/{result.trial_num}_cls.joblib.pkl"
        test_set_filenames = folder_path + f"/{result.trial_num}_testdata.joblib.pkl"
        _ = joblib.dump(result.optimal_classifier, cls_filename, compress=9)
        _ = joblib.dump(result.test_data, test_set_filenames, compress=9)

if __name__ == "__main__":
    run_all_trials()

