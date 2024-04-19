import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay ,cohen_kappa_score,precision_score

import cv2

BATCH_SIZE = 32
# model = tf.keras.models.load_model('./saved/best/Large/Batch 16 Split 0.8 No InnerMove/latest_checkpoint.h5')
model = tf.keras.models.load_model('./saved/latest_checkpoint.h5')
# model.summary()
# Visualize


test_dir  = './dataset/test'
# test_dir  = './dataset-b/test'
# test_dir  = '../../../../nodeserver/data/grades'
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


test_images = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=False,
                                                            batch_size=BATCH_SIZE,
                                                            # validation_split= 0.05,
                                                            # subset='validation',
                                                            seed=20,
                                                             label_mode = "categorical",
                                                            image_size=(224, 224))

class_names =  test_images.class_names
one_hot_true_labels = []
all_images = []
for images, labels in test_images:
    all_images.extend(images.numpy())
    one_hot_true_labels.extend(labels.numpy().astype(np.uint8))

true_labels = np.argmax(one_hot_true_labels, axis=1)
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in  next(zip(test_images)):
#         for i in range(9):
#                 ax = plt.subplot(3, 3, i + 1)
#                 plt.imshow(images[i].astype("uint8"))
#                 plt.title(class_names[np.argmax(labels[i])])
#                 plt.axis("off")
# plt.show()


def perf_measure(y_actual, y_pred):
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1


    return sum(TP), sum(FP), sum(TN), sum(FN)

# print('\nEvaluating test images....')
# test_loss, test_acc = model.evaluate(test_images, verbose=1)
# print('\nTest accuracy:', test_acc)
# print('\nPredicting test images....')
predictions = model.predict(test_images)

predicted_labels = np.argmax(predictions, axis=1)
# print(true_labels, predicted_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print( np.trace(conf_matrix) ,  np.sum(conf_matrix))
print('\nTest accuracy from confusion matrix:', overall_accuracy)


TP, FP, TN, FN = perf_measure(true_labels,predicted_labels)
print(TP, FP,TN,FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP)
false_positive_rate = FP / (TN + FP)



print("Confusion Matrix:")
print(conf_matrix)
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}")
print(f"Precision: {precision*100:.2f}")
print(f"Recall: {recall*100:.2f}")
print(f"F1 Score: {f1_score*100:.2f}")
print(f"Cohen Kappa: {cohen_kappa_score(true_labels, predicted_labels)*100:.2f}")
print(f"False Positive Rate: {false_positive_rate*100:.2f}")
print(f"Specificity: {specificity*100:.2f}")
correct_per_class = np.diag(conf_matrix)
total_per_class = np.sum(conf_matrix, axis=1)
incorrect_per_class = total_per_class - correct_per_class

accuracy_per_class = (correct_per_class / total_per_class) * 100

print("Accuracy Per Class: ", accuracy_per_class)

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
  #  np.argmax(one_hot_true_labels[i])==3 and 
  if np.argmax(one_hot_true_labels[i]) != np.argmax(predictions[i]) :
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

