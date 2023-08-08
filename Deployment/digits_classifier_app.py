from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("best_model.keras")

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
