import requests
import numpy as np
from tensorflow.keras.datasets import mnist
import json

# Constants
SAMPLES_NUMBER = 10

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be in the range [0, 1]
x_test = x_test.astype("float32") / 255.0

# Randomly select N samples from the test set
random_indices = np.random.choice(len(x_test), SAMPLES_NUMBER, replace=False)
sample_images = x_test[random_indices]
sample_labels = y_test[random_indices]

# Convert the images to the appropriate format and send them to the API one by one
url = "http://127.0.0.1:5000/predict"

# Predictions agregator
correct_predictions = 0

for i, image in enumerate(sample_images):
    # Convert the image to the format expected by the model (1, 28, 28, 1)
    image_reshaped = image.reshape(1, 28, 28, 1)

    # Convert the image array to JSON string
    data = json.dumps({"image": image_reshaped.tolist()})

    # Set the headers with the content type
    headers = {"content-type": "application/json"}

    # Send the POST request to the API
    response = requests.post(url, data=data, headers=headers)

# Process the API response
    if response.status_code == 200:
        result = response.json()
        predicted_digit = result.get("predicted_digit")
        actual_digit = sample_labels[i]
        print(f"Sample {i+1}: Predicted Digit: {predicted_digit}, Actual Digit: {actual_digit}")

        if predicted_digit == actual_digit:
            correct_predictions += 1
    else:
        print(f"Error occurred while making the API request for Sample {i+1}.")

accuracy = (correct_predictions / SAMPLES_NUMBER) * 100
print(f"Accuracy: {accuracy:.2f}%")
