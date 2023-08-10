# Libraries importing
import time
import subprocess
import multiprocessing
from multiprocessing import Process

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

    @step
    def start(self):
        print(100 * '#')
        print("!!! MAIN FLOW STARTED !!!")
        print(100 * '#')

        self.next(self.data_loading_step)

    @step
    def data_loading_step(self):
        self.train_set, self.test_set = data_loading()
        self.next(self.data_processing_step)

    @step
    def data_processing_step(self):
        self.x_train, self.x_test, self.y_train, self.y_test = data_processing(self.train_set, self.test_set)
        self.next(self.hyperparameter_optimization_step)

    @step
    def hyperparameter_optimization_step(self):
        self.best_model = hyperparameter_optimization(self.x_train, self.y_train, 
                                                      self.x_test, self.y_test, 
                                                      OPT_EPOCHS, BATCH_SIZE, 
                                                      VALIDATION_SPLIT)
        self.next(self.final_model_training_step)

    @step
    def final_model_training_step(self):
        self.final_model = final_model_training(self.best_model, self.x_train, 
                                                self.y_train, FINAL_EPOCHS, BATCH_SIZE, 
                                                VALIDATION_SPLIT)
        self.next(self.final_model_evaluation_step)

    @step
    def final_model_evaluation_step(self):
        final_model_evaluation(self.final_model, self.x_test, self.y_test)
        self.next(self.model_saving_step)

    @step
    def model_saving_step(self):
        model_saving(self.final_model, PATH, MODEL_NAME, FORMAT)
        self.next(self.api_testing_step)

    @step
    def api_testing_step(self):
        api_testing(SAMPLES_NUMBER)
        # subprocess.run(["python", "test_script.py"])
        self.next(self.end)

    @step
    def end(self):
        print(100 * '#')
        print("!!! MAIN FLOW ENDED !!!")
        print(100 * '#')

if __name__ == "__main__":
    MainFlow()

