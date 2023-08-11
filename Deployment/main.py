# Libraries importing
import subprocess
import concurrent.futures
import time
from task_duration import *

"""
This code snippet starts a Flask API in a separate process and runs a script 
called 'main_flow.py' using a thread pool executor. It then terminates 
the Flask API process and calculates the duration of the task.

"""

# Constants
PATH = "./models"
MODEL_NAME = "best_model"
FORMAT = "keras"

space = 35 * ' '

print(100 * '#')
print(100 * '#')
print(f"{space} !!! PROGRAM STARTED !!!")
print(100 * '#')
print(100 * '#')

start_time = time.time()

api_script = "api.py"
mlflow_script = "main_flow.py"
    
# Start the Flask API in a separate process
print()
print(100 * '#')
print("!!! API STARTED !!!")
print(100 * '#')

api_process = subprocess.Popen(["python", api_script, PATH, MODEL_NAME, FORMAT])

time.sleep(5)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.submit(subprocess.run(["python", mlflow_script, "run"]))

# Waiting for the test script to complete and then terminate the Flask API process
api_process.terminate()
api_process.wait()

print(100 * '#')
print(100 * '#')
print(f"{space} !!! PROGRAM ENDED !!!")
print(100 * '#')
print(100 * '#')

end_time = time.time()
task_duration(start_time, end_time)
