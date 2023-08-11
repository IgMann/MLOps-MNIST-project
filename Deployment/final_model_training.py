# Libraries loading
import time
import numpy as np
from task_duration import *
import tensorflow as tf
from tensorflow.keras.models import Sequential

def final_model_training(model: Sequential, x_train: np.ndarray, 
                        y_train: np.ndarray, epochs: int, batch_size: int, 
                        validation_split: float) -> Sequential:
    """
    Trains a given model using the provided training data.

    Args:
        model: The model to be trained.
        x_train: The input training data.
        y_train: The target training data.
        epochs: The number of epochs to train the model.
        batch_size: The batch size for training.
        validation_split: The validation split for training.

    Returns:
        The trained model.
    """

    print(100 * '=')
    print("FINAL MODEL TRAINING")
    print(100 * '=')

    start_time = time.time()

    print(100 * '-')
    print("Training started")
    print(100 * '-')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    print(100 * '-')
    print("Training finished")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)

    return model