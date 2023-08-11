# Libraries loading
import time
from typing import Tuple
import numpy as np
from task_duration import *
from tensorflow.keras.utils import to_categorical

def data_processing(train_set: Tuple[np.ndarray, np.ndarray], test_set: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the data by normalizing pixel values and performing one-hot encoding on the target labels.

    Args:
        train_set (Tuple[np.ndarray, np.ndarray]): A tuple containing the training images and labels.
        test_set (Tuple[np.ndarray, np.ndarray]): A tuple containing the test images and labels.

    Returns:
        x_train (np.ndarray): The normalized training images 
        x_test (np.ndarray): The normalized test images
        y_train (np.ndarray):  one-hot encoded training labels
        y_test (np.ndarray): one-hot encoded test labels
    """

    print(100 * '=')
    print("DATA PROCESSING")
    print(100 * '=')

    start_time = time.time()
    
    # Normalizing pixel values
    x_train = train_set[0].astype("float32") / 255.0
    x_test = test_set[0].astype("float32") / 255.0

    print(100 * '-')
    print("Normalizing pixel values finished")
    print(100 * '-')

    # One-hot encoding the target labels
    y_train = to_categorical(train_set[1], 10)
    y_test = to_categorical(test_set[1], 10)

    print(100 * '-')
    print("One-hot encoding the target labels finished")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)

    return x_train, x_test, y_train, y_test

