"""Used to perform different data ingestion steps for each file.
"""

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

test_length = lambda num_rows: min(int(0.2 * num_rows), num_rows - 5000) 

def electrical_grid_dataset() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    path = './datasets/elec_grid_stability.csv'
    elec = pd.read_csv(path, sep=',')

    X = elec.drop(['stab', 'stabf'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    Y = np.where(elec['stabf'] == 'stable', 1 ,0)

    X_train, X_test, Y_train, Y_test = train_test_split(Xsc, Y, train_size=5000, test_size=test_length(len(Y)))

    return (X_train, X_test, Y_train, Y_test)

def room_occupancy() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    path = './datasets/room_occupancy.csv'
    roomocc = pd.read_csv(path, sep=',')

    X = roomocc.drop(['number', 'date', 'Occupancy'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    Y = roomocc[['Occupancy']].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(Xsc, Y, train_size=5000, test_size=test_length(len(Y)))

    return (X_train, X_test, Y_train, Y_test)

def barbunya_beans():
    path = './datasets/Dry_Bean_Dataset.xlsx'
    beans = pd.read_excel(path)

    X = beans.drop(['Class'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    beans['Class'] = (beans['Class'] == 'BARBUNYA').astype(int)
    Y = beans[['Class']].values.ravel()

    return train_test_split(Xsc, Y, train_size=5000, test_size=test_length(len(Y)))


def stroke():
    """Classifier for those who have had strokes.

    Sourced from [Kaggle user fedesoriano](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
    """
    path = './datasets/stroke.csv'
    stroke_data = pd.read_csv(path, sep=',')
    X = stroke_data.drop(['id','stroke'], axis=1)
    to_one_hotify = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    X = pd.get_dummies(X, prefix=to_one_hotify, columns=to_one_hotify, drop_first=True)

    # bmi has some nan's, and is a [useless metric anyway](https://www.health.harvard.edu/blog/how-useful-is-the-body-mass-index-bmi-201603309339)
    X = X.drop(['bmi'], axis=1)


    Y = stroke_data[['stroke']].values.ravel()
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, train_size=5000, test_size=test_length(len(Y)))

    return (X_train, X_test, Y_train, Y_test)







ingestion_functions = [stroke, electrical_grid_dataset, room_occupancy, barbunya_beans]