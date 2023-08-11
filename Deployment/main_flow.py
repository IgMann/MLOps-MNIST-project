# Libraries importing
import time
import subprocess
import multiprocessing
from multiprocessing import Process
from typing import Tuple

from metaflow import FlowSpec, step

from task_duration import *
from data_loading import *
from data_processing import *
from hyperparameter_optimization import *
from final_model_training import *
from final_model_evaluation import *
from model_saving import *
from api_testing import *


# Constants
OPT_EPOCHS = 1
FINAL_EPOCHS = 1
BATCH_SIZE = 32 
VALIDATION_SPLIT = 0.1
MODEL_NAME = "best_model"
PATH = "./models"
MODEL_NAME = "best_model"
FORMAT = "keras"
SAMPLES_NUMBER = 10

# Main flow 
class MainFlow(FlowSpec):
    """
    MainFlow class is a Metaflow flow that performs a series of steps to train and evaluate a machine learning model.

    The flow consists of the following steps:
    1. start: Entry point of the flow. Prints a message indicating that the main flow has started.
    2. data_loading_step: Loads the training and test datasets.
    3. data_processing_step: Processes the loaded datasets by splitting them into input features and target labels.
    4. hyperparameter_optimization_step: Performs hyperparameter optimization on a model using the processed datasets.
    5. final_model_training_step: Trains the final model using the best model obtained from the hyperparameter optimization step and the processed datasets.
    6. final_model_evaluation_step: Evaluates the performance of the final model using the test dataset.
    7. model_saving_step: Saves the final model to a specified path with a specified name and format.
    8. api_testing_step: Tests the API using a specified number of samples.
    9. end: Prints a message indicating that the main flow has ended.
    """

    def __init__(self):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.final_model = None

    @step
    def start(self):
        """
        Entry point of the flow. Prints a message indicating that the main flow has started.
        """
        print(100 * '#')
        print("!!! MAIN FLOW STARTED !!!")
        print(100 * '#')
        self.next(self.data_loading_step)

    @step
    def data_loading_step(self):
        """
        Loads the training and test datasets.
        """
        self.train_set, self.test_set = data_loading()
        self.next(self.data_processing_step)

    @step
    def data_processing_step(self):
        """
        Processes the loaded datasets by splitting them into input features and target labels.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = data_processing(self.train_set, self.test_set)
        self.next(self.hyperparameter_optimization_step)

    @step
    def hyperparameter_optimization_step(self):
        """
        Performs hyperparameter optimization on a model using the processed datasets.
        """
        self.best_model = hyperparameter_optimization(self.x_train, self.y_train, 
                                                      self.x_test, self.y_test, 
                                                      OPT_EPOCHS, BATCH_SIZE, 
                                                      VALIDATION_SPLIT)
        self.next(self.final_model_training_step)

    @step
    def final_model_training_step(self):
        """
        Trains the final model using the best model obtained from the hyperparameter optimization step and the processed datasets.
        """
        self.final_model = final_model_training(self.best_model, self.x_train, 
                                                self.y_train, FINAL_EPOCHS, BATCH_SIZE, 
                                                VALIDATION_SPLIT)
        self.next(self.final_model_evaluation_step)

    @step
    def final_model_evaluation_step(self):
        """
        Evaluates the performance of the final model using the test dataset.
        """
        final_model_evaluation(self.final_model, self.x_test, self.y_test)
        self.next(self.model_saving_step)

    @step
    def model_saving_step(self):
        """
        Saves the final model to a specified path with a specified name and format.
        """
        model_saving(self.final_model, PATH, MODEL_NAME, FORMAT)
        self.next(self.api_testing_step)

    @step
    def api_testing_step(self):
        """
        Tests the API using a specified number of samples.
        """
        api_testing(SAMPLES_NUMBER)
        # subprocess.run(["python", "test_script.py"])
        self.next(self.end)

    @step
    def end(self):
        """
        Prints a message indicating that the main flow has ended.
        """
        print(100 * '#')
        print("!!! MAIN FLOW ENDED !!!")
        print(100 * '#')

if __name__ == "__main__":
    MainFlow()

