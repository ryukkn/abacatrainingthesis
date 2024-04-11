import numpy as np
import tensorflow as tf
import os

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./saved/model.tflite")
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

# Test images directory
test_images_dir = "./dataset/test"
accuracy = []
# Iterate through test images
index = 0
for folder in os.listdir(test_images_dir):
    count = 0
    length = len(os.listdir(os.path.join(test_images_dir, folder)))
    correct = 0.0
    for filename in os.listdir(os.path.join(test_images_dir, folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(test_images_dir, os.path.join(folder, filename))
            
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
            predictions = np.argmax(output_data, axis=1)  # Get index of the maximum value
            if(predictions[0] == index):
                correct+=1
            count+=1
            # print("Image:", filename, "Predicted class:", predictions)
    accuracy.append(correct/length)
    index +=1

print(accuracy)