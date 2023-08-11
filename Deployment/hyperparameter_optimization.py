# Libraries loading
import time
import numpy as np
from task_duration import *
from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from hyperopt import fmin, tpe, hp
import mlflow.tensorflow

def hyperparameter_optimization(x_train: np.ndarray,
                                y_train: np.ndarray,
                                x_test: np.ndarray,
                                y_test: np.ndarray,
                                num_epochs: int,
                                batch_size: int,
                                validation_split: float) -> Sequential:
    """
    Perform hyperparameter optimization for a neural network model using the Hyperopt library.

    Args:
        x_train: The training data.
        y_train: The training labels.
        x_test: The testing data.
        y_test: The testing labels.
        num_epochs: The number of epochs to train the model.
        batch_size: The batch size for training.
        validation_split: The fraction of the training data to be used for validation.

    Returns:
        The best model found during the optimization process.
    """

    mlflow.tensorflow.autolog()

    def build_model(parameters: Dict[str, Any]) -> Sequential:
        """
        Build a neural network model with the given hyperparameters.

        Args:
            parameters: The hyperparameters for building the model.

        Returns:
            The built model.
        """
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))

        for units in parameters["units"]:
            model.add(Dense(units, activation="relu"))

        model.add(Dense(10, activation="softmax"))

        optimizer = parameters["optimizer"]
        model.compile(optimizer=optimizer(learning_rate=parameters["learning_rate"]),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    best_accuracy = 0.0
    best_model = None

    def objective(parameters: Dict[str, Any]) -> float:
        """
        Evaluate the performance of a model with the given hyperparameters.

        Args:
            parameters: The hyperparameters for building and evaluating the model.

        Returns:
            The negative validation accuracy of the model.
        """
        nonlocal best_accuracy, best_model
        model = build_model(parameters)
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                            validation_split=validation_split, verbose=0)
        val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model  # Save the best model

        return -val_accuracy

    space = {
        "num_hidden_layers": hp.choice("num_hidden_layers", [1, 2, 3]),
        "units": hp.choice("units", [[64], [128], [256]]),
        "optimizer": hp.choice("optimizer", [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam]),
        "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01)
    }

    print(100 * '=')
    print("HYPERPARAMETER OPTIMIZATION")
    print(100 * '=')

    start_time = time.time()

    print(100 * '-')
    print("Hyperparameter optimization started")
    print(100 * '-')

    best_parameters = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)

    print(100 * '-')
    print("Hyperparameter optimization ended")
    print(100 * '-')

    best_num_hidden_layers = [1, 2, 3][best_parameters["num_hidden_layers"]]
    best_units = [[64], [128], [256]][best_parameters["units"]]
    best_optimizer = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam][best_parameters["optimizer"]]
    best_learning_rate = best_parameters["learning_rate"]

    print(100 * '-')
    print("Best Hyperparameters:")
    print(100 * '-')

    print(f"Number of Hidden Layers: {best_num_hidden_layers}")
    print(f"Units in Hidden Layers: {best_units}")
    print(f"Optimizer: {best_optimizer.__name__}")
    print(f"Learning Rate: {best_learning_rate}")
    print(100 * "-")

    end_time = time.time()
    task_duration(start_time, end_time)

    return best_model