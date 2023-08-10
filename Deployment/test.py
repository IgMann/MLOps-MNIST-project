import os
from datetime import datetime
from tensorflow.keras.models import load_model

path = "./models"
format = "keras"
print(os.listdir(path))
model_files = [filename for filename in os.listdir(path) if filename.endswith(f'.{format}')]
def get_datetime_from_filename(filename):
    timestamp = filename.split('_')[-1].split('.')[0]
    return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")

newest_model_filename = max(model_files, key=get_datetime_from_filename)

model = load_model(newest_model_filename)

print(model_files)
print(newest_model_file)