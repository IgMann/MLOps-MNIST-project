# Libraries loading
import time
import numpy as np
from task_duration import *
import tensorflow as tf
from tensorflow.keras.models import Sequential

def final_model_evaluation(model: Sequential, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the performance of a given model on a test dataset.

    Args:
        model: The model to be evaluated.
        x_test: The test data.
        y_test: The test labels.

    Returns:
        None
    """
    print(100 * '=')
    print("FINAL MODEL EVALUATION")
    print(100 * '=')

    start_time = time.time()

    loss, accuracy = model.evaluate(x_test, y_test)

    print(100 * '-')
    print(f"Test accuracy with best hyperparameters: {accuracy * 100:.2f}%")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)