from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import GridSearchCV, validation_curve
import matplotlib.pyplot as plt
import numpy as np
import time

def run_neuralNetwork(X_train, X_test, y_train, y_test, dataset_name):
    print("Running Neural Network for ", dataset_name)

    classification_objective = MLPClassifier
    custom_tuning_parameters = { 'learning_rate_init': [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
    custom_parameters = {'max_iter': 5000, 'hidden_layer_sizes': (5, 2), 'activation': 'logistic', 'verbose': False}

    # Base accuracy - https://sachinkmr375.medium.com/testing-the-models-accuracy-in-ml-8385ee944e3f
    classification = classification_objective(**custom_parameters)
    classification.fit(X_train, y_train)
    y_pred = classification.predict(X_test)
    initial_score = accuracy_score(y_test, y_pred)

    # Validation Curve - https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    cross_validation = 5
    scoring = 'accuracy'
    for key, value in custom_tuning_parameters.items():
        if key == 'hidden_layer_sizes':
            continue
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
    visualization = LearningCurve(classification, cv=cross_validation, scoring=scoring, train_sizes=sizes, n_jobs=8)
    visualization.fit(X_train, y_train)
    visualization.show("results/Neural_Network_learning_curve_{}.png".format(dataset_name))
    plt.clf()

    final_classification = classification_objective()
    final_classification.set_params(**custom_parameters)
    final_classification.set_params(**tuned_classification.best_params_)
    final_classification.fit(X_train, y_train)

    # 6. Calculate test score and record query time
    start_time = time.time()
    y_pred = final_classification.predict(X_test)
    query_time = time.time() - start_time

    # Final metrics
    start_time = time.time()
    y_pred = final_classification.predict(X_test)
    query_time = time.time() - start_time
    final_score = accuracy_score(y_test, y_pred)
    print("Query time: {}".format(query_time))
    print("Before Tuned Score: {}".format(initial_score))
    print("After Tuned Score: {}".format(final_score))

    generate_nn_curves(final_classification, dataset_name, final_classification.loss_curve_, X_train, y_train, X_test, y_test)

def generate_validation_curve(model,  param_name, param_range, scoring, cv, dataset_name, X_train, y_train):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Neural Network")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.semilogx(param_range, train_scores_mean, label="Training score")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", marker='o', color="#9fc377")
    plt.legend(loc="best")
    plt.savefig("results/Neural_Network_model_complexity_{}_{}.png".format( dataset_name, param_name))
    plt.clf()

def generate_nn_curves(classification, dataset_name, loss_curve, X_train, y_train, X_test, y_test):
    # https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
    plt.title("Loss Curve for {}".format(dataset_name))
    plt.xlabel("epoch")
    plt.plot(loss_curve)
    plt.savefig("results/Neural_Network_loss_curve_{}.png".format(dataset_name))
    plt.clf()

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    scores_train = []
    scores_test = []

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 25
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    classification = MLPClassifier(**classification.get_params())

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            classification.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(classification.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(classification.score(X_test, y_test))

        epoch += 1

    """ Plot """
    plt.plot(scores_train, alpha=0.8, label="Training score")
    plt.plot(scores_test, alpha=0.8, label="Cross-validation score")
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig("results/Neural_Network_accuracy_over_epochs_{}.png".format(dataset_name))
    plt.clf()