# Libraries loading
import os
import time
from task_duration import *
import tensorflow as tf
from datetime import datetime

def model_saving(model, path, model_name, format):
    print(100 * '=')
    print("FINAL MODEL SAVING")
    print(100 * '=')

    start_time = time.time()

    if not os.path.exists(path):
        os.makedirs(path)

    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model.save(f"{path}/{model_name}_{current_datetime}.{format}")

    print(100 * '-')
    print("Model saved")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)