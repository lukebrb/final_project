"""Used to perform different data ingestion steps for each file.
"""

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def electrical_grid_dataset() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    path = './datasets/elec_grid_stability.csv'
    elec = pd.read_csv(path, sep=',')

    X = elec.drop(['stab', 'stabf'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    Y = elec[['stabf']].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(Xsc, Y, train_size=5000)

    return(X_train, X_test, Y_train, Y_test)

def room_occupancy() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    path = './datasets/room_occupancy.csv'
    roomocc = pd.read_csv(path, sep=',')

    X = roomocc.drop(['number', 'date', 'Occupancy'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    Y = roomocc[['Occupancy']].values.ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(Xsc, Y, train_size=5000)

    return(X_train, X_test, Y_train, Y_test)

def barbunya_beans():
    path = './datasets/Dry_Bean_Dataset.xlsx'
    beans = pd.read_excel(path)

    X = beans.drop(['Class'], axis=1)
    Xsc = StandardScaler().fit(X).transform(X)
    beans['Class'] = (beans['Class'] == 'BARBUNYA').astype(int)
    Y = beans[['Class']].values.ravel()

    return train_test_split(Xsc, Y, train_size=5000)




ingestion_functions = [electrical_grid_dataset, room_occupancy, barbunya_beans]