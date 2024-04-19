import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay 

import cv2
model = tf.keras.models.load_model('./saved/best/Small/Batch 16 split 0.8/latest_checkpoint.h5')
model.summary()
# Visualize

def foreground_extractor(x):
     # Convert image to grayscale
    gray = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Apply adaptive thresholding to separate foreground from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with white background
    mask = np.ones_like(x) * 255

    # Draw filled contours on the mask
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Bitwise AND operation to get the foreground
    x = cv2.bitwise_and(x, mask)

    # threshold = 100  # Adjust as needed

    # Convert pixels close to black to white
    # x[np.sum(x < threshold, axis=2) == 3] = [255, 255, 255]

    return x.astype(np.float64)

def crop_center(image, crop_width, crop_height):
    """
    Crop the center portion of the image with specified width and height.
    """
    height, width = image.shape[:2]
    
    # Calculate starting and ending indices for cropping
    start_x = max(0, (width - crop_width) // 2)
    start_y = max(0, (height - crop_height) // 2)
    end_x = min(width, start_x + crop_width)
    end_y = min(height, start_y + crop_height)
    
    # Perform cropping
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cv2.resize(cropped_image, (224,224))

def crop():
    def _crop(x):
    #     height, width = x.shape[:2]
    #     return  crop_center(x, int(width/1.5), int(width/1.5))
          # return ((x/127.5)-1)
          return x
    return _crop

def plot_image(i, predictions_array, true_label, img):
  true_label, img = np.argmax(true_label[i]), img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  # plt.imshow(img, cmap=plt.cm.binary)s
  # print(np.array(img).shape)
  plt.imshow(img.astype("uint8"))

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = np.argmax(true_label[i])
  plt.grid(False)
  plt.xticks(range(len(class_names)))
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


test_dir  = './dataset/test'
test_datagen = ImageDataGenerator(
                                    preprocessing_function  = crop(),
     
                                   )
                                   

test_images = test_datagen.flow_from_directory(test_dir, target_size=(224,224),
                batch_size=32,class_mode="categorical", shuffle=False)

class_names = list(test_images.class_indices.keys())
true_labels = test_images.classes
all_images = []
for i in range(len(test_images)):
    batch_images, _ = test_images[i]
    all_images.extend(batch_images)
one_hot_true_labels = np.eye(len(class_names))[true_labels]
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in  next(zip(test_images)):
#         for i in range(9):
#                 ax = plt.subplot(3, 3, i + 1)
#                 plt.imshow(images[i].astype("uint8"))
#                 plt.title(class_names[np.argmax(labels[i])])
#                 plt.axis("off")
# plt.show()

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# print('\nEvaluating test images....')
# test_loss, test_acc = model.evaluate(test_images, verbose=1)
# print('\nTest accuracy:', test_acc)
print('\nPredicting test images....')
predictions = model.predict(test_images)

predicted_labels = np.argmax(predictions, axis=1)
# print(true_labels, predicted_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print('\nTest accuracy from confusion matrix:', overall_accuracy)
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)

print(FP, TN, FN, TP)
FP, FN, TP, TN = sum(FP), sum(FN), sum(TP), sum(TN)
print(FP, TN, FN, TP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP)
false_positive_rate = FP / (TN + FP)



print("Confusion Matrix:")
print(conf_matrix)
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
correct_per_class = np.diag(conf_matrix)
total_per_class = np.sum(conf_matrix, axis=1)
incorrect_per_class = total_per_class - correct_per_class

plt.figure(figsize=(10, 6))
plt.bar(class_names, correct_per_class, color='green', label='Correct Predictions')
plt.bar(class_names, incorrect_per_class, bottom=correct_per_class, color='red', label='Incorrect Predictions')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Correct Predictions Per Class')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=class_names)
disp.plot()
plt.show()


# test_images,test_labels = next(zip(test_images))[0]
# _test_images,test_labels = zip(*test_images)
# print(len(test_images))
# num_rows = math.floor(math.sqrt(len(test_labels)))
# num_cols = math.floor(math.sqrt(len(test_labels)))
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
j = 0
i = 0
while j < num_images:
  if np.argmax(one_hot_true_labels[i])==3 and np.argmax(one_hot_true_labels[i]) != np.argmax(predictions[i]) :
    plt.subplot(num_rows, 2*num_cols, 2*j+1)
    plot_image(i, predictions[i], one_hot_true_labels, all_images)
    plt.subplot(num_rows, 2*num_cols, 2*j+2)
    plot_value_array(i, predictions[i], one_hot_true_labels)
    j+=1
  i+=1
  if(i >= len(predictions)):
      break
  
plt.tight_layout()
plt.show()

#Predict ONE
# img_width, img_height = 224, 224
# img = tf.keras.utils.load_img('../../../../nodeserver/data/grades/S2/S2-0039.jpg', target_size = (img_width, img_height))
# img = tf.keras.utils.img_to_array(img)
# img = np.expand_dims(img, axis = 0)
# prediction = model.predict(img)

