from typing import Any
# Libraries loading
import os
import time
from task_duration import *
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential

def model_saving(model: Sequential, path: str, model_name: str, format: str) -> None:
    """
    Save a machine learning model to a specified path with a given name and format.
    
    Args:
        model: The machine learning model to be saved.
        path: The path where the model will be saved.
        model_name: The name of the model.
        format: The format in which the model will be saved.
    """
    print(100 * '=')
    print("FINAL MODEL SAVING")
    print(100 * '=')

    start_time = time.time()

    os.makedirs(path, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model.save(f"{path}/{model_name}_{current_datetime}.{format}")

    print(100 * '-')
    print("Model saved")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)