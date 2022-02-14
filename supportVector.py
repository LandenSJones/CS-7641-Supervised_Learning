from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import GridSearchCV, validation_curve
import matplotlib.pyplot as plt
import numpy as np
import time

# https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
def run_supportVector_rbf(X_train, X_test, y_train, y_test, dataset_name, kernal = 'rbf'):
    print("Running Support Vector Machines for ", dataset_name)

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    classification_objective = SVC
    custom_tuning_parameters = {'C': np.logspace(-5, 5, 11)}
    custom_parameters = {'kernel': 'rbf', 'verbose': False}

    # Base accuracy - https://sachinkmr375.medium.com/testing-the-models-accuracy-in-ml-8385ee944e3f
    clf = classification_objective(**custom_parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    initial_score = accuracy_score(y_test, y_pred)

    # Validation Curve - https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    cross_validation = 10
    scoring = 'accuracy'
    for key, value in custom_tuning_parameters.items():
        generate_validation_curve(classification_objective(**custom_parameters), key, value, scoring, cross_validation, dataset_name, X_train, y_train, kernal)

    # Grid search - https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
    classification = classification_objective(**custom_parameters)
    grid_params = [custom_tuning_parameters]
    tuned_classification = GridSearchCV(classification, grid_params, scoring=scoring, cv=cross_validation, verbose=0)
    tuned_classification.fit(X_train, y_train)

    print("Best parameters found: ", tuned_classification.best_params_)

    # Tuned classifier curve generation - https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
    # https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    classification = classification_objective(**custom_parameters)
    classification.set_params(**tuned_classification.best_params_)
    sizes = np.linspace(0.1, 1.0, 10)
    visualization = LearningCurve(classification, cv=cross_validation, scoring=scoring, train_sizes=sizes, n_jobs=8)
    visualization.fit(X_train, y_train)
    visualization.show("results/SVM_learning_curve_{}_{}.png".format(dataset_name, kernal))
    plt.clf()

    final_classification = classification_objective()
    final_classification.set_params(**custom_parameters)
    final_classification.set_params(**tuned_classification.best_params_)
    final_classification.fit(X_train, y_train)

    # Final metrics
    start_time = time.time()
    y_pred = final_classification.predict(X_test)
    query_time = time.time() - start_time
    final_score = accuracy_score(y_test, y_pred)
    print("Query time: {}".format(query_time))
    print("Before Tuned Score: {}".format(initial_score))
    print("After Tuned Score: {}".format(final_score))

def run_supportVector_sigmoid(X_train, X_test, y_train, y_test, dataset_name, kernal = 'sigmoid'):
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    classification_objective = SVC
    custom_tuning_parameters = {'C': np.logspace(-5, 5, 11)}
    custom_parameters = {'kernel': 'sigmoid', 'verbose': False}

    # Base accuracy - https://sachinkmr375.medium.com/testing-the-models-accuracy-in-ml-8385ee944e3f
    classification = classification_objective(**custom_parameters)
    classification.fit(X_train, y_train)
    y_pred = classification.predict(X_test)
    initial_score = accuracy_score(y_test, y_pred)

    # Validation Curve - https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    cross_validation = 10
    scoring = 'accuracy'
    for key, value in custom_tuning_parameters.items():
        generate_validation_curve(classification_objective(**custom_parameters), key, value, scoring, cross_validation, dataset_name, X_train, y_train, kernal)

    # Grid search - https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
    classification = classification_objective(**custom_parameters)
    grid_params = [custom_tuning_parameters]
    tuned_classification = GridSearchCV(classification, grid_params, scoring=scoring, cv=cross_validation, verbose=0)
    tuned_classification.fit(X_train, y_train)

    print("Best parameters found: ", tuned_classification.best_params_)

    # Tuned classifier curve generation - https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
    # https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    classification = classification_objective(**custom_parameters)
    classification.set_params(**tuned_classification.best_params_)
    sizes = np.linspace(0.1, 1.0, 10)
    visualization = LearningCurve(classification, cv=cross_validation, scoring=scoring, train_sizes=sizes, n_jobs=8)
    visualization.fit(X_train, y_train)
    visualization.show("results/SVM_learning_curve_{}_{}.png".format(dataset_name, kernal))
    plt.clf()

    final_classification = classification_objective()
    final_classification.set_params(**custom_parameters)
    final_classification.set_params(**tuned_classification.best_params_)
    final_classification.fit(X_train, y_train)

    # Final metrics
    start_time = time.time()
    y_pred = final_classification.predict(X_test)
    query_time = time.time() - start_time
    final_score = accuracy_score(y_test, y_pred)
    print("Query time: {}".format(query_time))
    print("Before Tuned Score: {}".format(initial_score))
    print("After Tuned Score: {}".format(final_score))

def run_supportVector(X_train, X_test, y_train, y_test, dataset_name):
    run_supportVector_sigmoid(X_train, X_test, y_train, y_test, dataset_name)
    run_supportVector_rbf(X_train, X_test, y_train, y_test, dataset_name)

def generate_validation_curve(model, param_name, param_range, scoring, cv, dataset_name, X_train, y_train, kernal):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.semilogx(param_range, train_scores_mean, label="Training score", marker='o', color="#0272a2")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", marker='o', color="#9fc377")
    plt.legend(loc="best")
    plt.savefig("results/SVM_model_complexity_{}_{}_{}.png".format(dataset_name, param_name, kernal))
    plt.clf()