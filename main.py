from boosting import run_boosting
from decisionTree import run_decision_tree
from knn import run_knn
from neuralNetwork import run_neuralNetwork
from supportVector import run_supportVector

from multiprocessing import Process

import pandas as pd

from sklearn.model_selection import train_test_split

def run_experiments():
    datasets = [ 'new-thyroid.data', 'wine.data']
    for dataset in datasets:
        dataset_name = dataset.split('.')[0]
        X_train, X_test, y_train, y_test = load_and_train_data(dataset)
        run_boosting(X_train, X_test, y_train, y_test, dataset_name)
        run_decision_tree(X_train, X_test, y_train, y_test, dataset_name)
        run_knn(X_train, X_test, y_train, y_test, dataset_name)
        run_neuralNetwork(X_train, X_test, y_train, y_test, dataset_name)
        run_supportVector(X_train, X_test, y_train, y_test, dataset_name)

def load_and_train_data(dataset):
    df = pd.read_csv("Data/" + dataset)
    df.dropna()
    X, y = df.iloc[:, :-1], df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    return X_train, X_test, y_train, y_test

def main():
    run_experiments()

if __name__ == "__main__":
    main()
