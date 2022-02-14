from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import time

def run_decision_tree(X_train, X_test, y_train, y_test, dataset_name):
    print("Running Decision Tree for ", dataset_name)

    classification_objective = DecisionTreeClassifier
    custom_tuning_parameters = {'max_depth': range(1, 50), 'min_samples_leaf': range(1, 20)}
    custom_parameters = {}

    # Base accuracy - https://sachinkmr375.medium.com/testing-the-models-accuracy-in-ml-8385ee944e3f
    classification = classification_objective(**custom_parameters)
    classification.fit(X_train, y_train)
    y_pred = classification.predict(X_test)
    initial_score = accuracy_score(y_test, y_pred)

    # Validation Curve - https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    cross_validation = 2
    scoring = 'accuracy'
    for key, value in custom_tuning_parameters.items():
        generate_validation_curve(classification_objective(**custom_parameters), key, value, scoring, cross_validation, dataset_name, X_train, y_train)

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
    visualization = LearningCurve(classification, cv=cross_validation, scoring=scoring, train_sizes=sizes, n_jobs=5)
    visualization.fit(X_train, y_train)
    visualization.show("results/DecisionTree_learning_curve_{}.png".format(dataset_name))
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

def generate_validation_curve(model, param_name, param_range, scoring, cv, dataset_name, X_train, y_train):
    viz = ValidationCurve(model, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
    viz.fit(X_train, y_train)
    viz.show("results/DecisionTree_model_complexity_{}_{}.png".format(dataset_name, param_name))
    plt.clf()