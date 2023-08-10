# Libraries importing
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

def newest_model(path, model_name, format):
    model_files = [filename for filename in os.listdir(path) if filename.endswith(f'.{format}')]
    
    def get_datetime_from_filename(filename):
        timestamp = filename.split('_')[-1].split('.')[0]
        return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")

    newest_model_filename = max(model_files, key=get_datetime_from_filename)

    return newest_model_filename

app = Flask(__name__)

# Load constants
PATH = sys.argv[1]
MODEL_NAME = sys.argv[2]
FORMAT = sys.argv[3]

# Load the trained model
try:
    if flag:
        pass
except:
    flag = False
    global newest_model_filename
    newest_model_filename = newest_model(PATH, MODEL_NAME, FORMAT)
    
    global model
    model = load_model(f"{PATH}/{newest_model_filename}")

    print(100 * '-')
    print(f"Model {newest_model_filename} loaded")
    print(100 * '-')

# API endpoint to reload the model
@app.route("/reload_model", methods=["GET"])
def reload_model():
    global newest_model_filename
    old_model_filename = newest_model_filename
    newest_model_filename = newest_model(PATH, MODEL_NAME, FORMAT)
    model = load_model(f"{PATH}/{newest_model_filename}")

    if old_model_filename != newest_model_filename:
        print(100 * '-')
        print(f"Model sucessfully upgraded!")
        print(f"Old model: {old_model_filename}")
        print(f"New model: {newest_model_filename}")
        print(100 * '-')
    else:
        print(100 * '-')
        print(f"Model sucessfully reloaded!")
        print(100 * '-')

    return jsonify({"message": "Model reloaded successfully"}), 200

# API endpoint to predict the digit in the image
@app.route("/predict", methods=["POST"])
def predict_digit():
    if request.method == "POST":
        # Get the JSON data from the request
        data = request.get_json()

        # Extract and preprocess the image data
        image_data = np.array(data["image"]).reshape(28, 28, 1)

        # Make the prediction
        prediction = model.predict(np.expand_dims(image_data, axis=0))
        predicted_digit = np.argmax(prediction[0])

        return jsonify({"predicted_digit": int(predicted_digit)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
