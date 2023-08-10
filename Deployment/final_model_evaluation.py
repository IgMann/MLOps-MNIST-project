# Libraries loading
import time
from task_duration import *
import tensorflow as tf

def final_model_evaluation(model, x_test, y_test):
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