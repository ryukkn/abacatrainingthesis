import numpy as np
import tensorflow as tf
import os

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='./saved/model.tflite')
# interpreter = tf.lite.Interpreter(model_path='../../../../abaca_v1/abaca_classification/assets/model/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image
def preprocess_image(image_path):
    # Load and preprocess the image as required by your model
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Ensure 3 channels (RGB)
    image = tf.image.resize(image, (224, 224))  # Resize to match input size of your model
    # image = (image / 127.5)-1  # Normalize pixel values to [-1, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

image_path = "./g3.jpg"

# Preprocess image
input_data = preprocess_image(image_path)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Post-processing or evaluation based on your model's output
# (e.g., softmax for classification, argmax for prediction, etc.).

# Example:
print("Result:", np.around(output_data[0], decimals=4))
predictions = np.argmax(output_data, axis=1)[0]  # Get index of the maximum value
print("Predition:", predictions)
print("Confidence:", output_data[0][predictions]*100)
