# Libraries loading
import time
from task_duration import *
import tensorflow as tf

def final_model_training(model, x_train, y_train, epochs, batch_size, validation_split):
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