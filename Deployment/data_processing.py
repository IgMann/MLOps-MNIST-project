# Libraries loading
import time
from task_duration import *
from tensorflow.keras.utils import to_categorical

def data_processing(train_set, test_set):
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

