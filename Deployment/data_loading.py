# Libraries loading
import time
from task_duration import *
from keras.datasets import mnist

def data_loading():
    print(100 * '=')
    print("DATA LOADING")
    print(100 * '=')

    start_time = time.time()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
    print(100 * '-')
    print("Train and test set shapes:")
    print(100 * '-')
    print(f"x_train set shape: {x_train.shape}")
    print(f"y_train set shape: {y_train.shape}")
    print(f"x_test set shape: {x_test.shape}")
    print(f"y_test set shape: {y_test.shape}")
    print(100 * '-')

    end_time = time.time()
    task_duration(start_time, end_time)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    data_loading()
