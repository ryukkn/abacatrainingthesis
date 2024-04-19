from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.metrics import CategoricalAccuracy
from PIL import Image
from PIL import ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True # needed for working with this dataset

from random import sample
def geodesic_reconstruction_MMCE(image, structuring_element_radius, num_iterations):
    # Define structuring element as a disk
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_element_radius, structuring_element_radius))
    
    # Initialize enhanced image
    enhanced_image = np.copy(image).astype(np.float32)
    
    # Convert the image to float32
    enhanced_image = enhanced_image.astype(np.float32)
    
    # Split the image into color channels
    b, g, r = cv2.split(enhanced_image)
    
    # Perform iterations for each color channel
    for i in range(num_iterations):
        # Calculate top-hat transform by reconstruction for each channel
        opening_b = cv2.morphologyEx(b, cv2.MORPH_OPEN, structuring_element)
        top_hat_b = b - opening_b
        
        closing_b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, structuring_element)
        bottom_hat_b = closing_b - b
        
        opening_g = cv2.morphologyEx(g, cv2.MORPH_OPEN, structuring_element)
        top_hat_g = g - opening_g
        
        closing_g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, structuring_element)
        bottom_hat_g = closing_g - g
        
        opening_r = cv2.morphologyEx(r, cv2.MORPH_OPEN, structuring_element)
        top_hat_r = r - opening_r
        
        closing_r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, structuring_element)
        bottom_hat_r = closing_r - r
        
        # Calculate subtraction from neighboring scales for each channel
        if i > 0:
            top_hat_sub_b = top_hat_b - prev_top_hat_b
            bottom_hat_sub_b = bottom_hat_b - prev_bottom_hat_b
            top_hat_sub_g = top_hat_g - prev_top_hat_g
            bottom_hat_sub_g = bottom_hat_g - prev_bottom_hat_g
            top_hat_sub_r = top_hat_r - prev_top_hat_r
            bottom_hat_sub_r = bottom_hat_r - prev_bottom_hat_r
        else:
            top_hat_sub_b = top_hat_b
            bottom_hat_sub_b = bottom_hat_b
            top_hat_sub_g = top_hat_g
            bottom_hat_sub_g = bottom_hat_g
            top_hat_sub_r = top_hat_r
            bottom_hat_sub_r = bottom_hat_r
        
        # Update previous top-hat and bottom-hat for next iteration
        prev_top_hat_b = top_hat_b
        prev_bottom_hat_b = bottom_hat_b
        prev_top_hat_g = top_hat_g
        prev_bottom_hat_g = bottom_hat_g
        prev_top_hat_r = top_hat_r
        prev_bottom_hat_r = bottom_hat_r
        
        # Calculate maximum values of all scales obtained for each channel
        if i == 0:
            max_top_hat_b = top_hat_b
            max_bottom_hat_b = bottom_hat_b
            max_top_hat_g = top_hat_g
            max_bottom_hat_g = bottom_hat_g
            max_top_hat_r = top_hat_r
            max_bottom_hat_r = bottom_hat_r
        else:
            max_top_hat_b = np.maximum(max_top_hat_b, top_hat_b)
            max_bottom_hat_b = np.maximum(max_bottom_hat_b, bottom_hat_b)
            max_top_hat_g = np.maximum(max_top_hat_g, top_hat_g)
            max_bottom_hat_g = np.maximum(max_bottom_hat_g, bottom_hat_g)
            max_top_hat_r = np.maximum(max_top_hat_r, top_hat_r)
            max_bottom_hat_r = np.maximum(max_bottom_hat_r, bottom_hat_r)
        
    # Calculate contrast enhancement for each channel
    enhanced_b = b + (max_top_hat_b + top_hat_sub_b) - (max_bottom_hat_b + bottom_hat_sub_b)
    enhanced_g = g + (max_top_hat_g + top_hat_sub_g) - (max_bottom_hat_g + bottom_hat_sub_g)
    enhanced_r = r + (max_top_hat_r + top_hat_sub_r) - (max_bottom_hat_r + bottom_hat_sub_r)
    
    # Merge the color channels back into a single image
    enhanced_image = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
    
    # Convert the enhanced image back to uint8
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image

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


    # Convert pixels close to black to white
    threshold = 100  # Adjust as needed
    x[np.sum(x < threshold, axis=2) == 3] = [255, 255, 255]
    return x.astype(np.float64)
# , 96  128, 156
def preprocess_functions(p=1.0, size_list=[64, 72, 96], augmented=False):
    def _preprocess_functions(x):
        height, width = x.shape[:2]
        # x = foreground_extractor(x)
        # x = crop_center(x, int(width/1.5), int(width/1.5))
        # Inner move
        x = crop()(x)
        if(np.random.random() < p) and augmented:
            # x =  geodesic_reconstruction_mmce(x)
            _patch_size = sample(size_list, k=1)[0]
            patch_size = (_patch_size, _patch_size)
            # random offset
            offsetx = np.random.randint(1, 10)
            offsety = np.random.randint(2, height-_patch_size) - 1
            # Swap two patches
            patch_1 = x[offsety:patch_size[0]+offsety, offsetx:patch_size[1]+offsetx].copy()
            patch_2 = x[-(patch_size[0]+offsety):-offsety, -(patch_size[1]+offsetx):-offsetx].copy()
            # Swap pixels between patches
            x[offsety:patch_size[0]+offsety, offsetx:patch_size[1]+offsetx] = patch_2
            x[-(patch_size[0]+offsety):-offsety, -(patch_size[1]+offsetx):-offsetx] = patch_1
        # return ((x/127.5)-1) 

        
        return x
    return _preprocess_functions



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
    
    # return    foreground_extractor(cv2.resize(cropped_image, (224,224),  interpolation=cv2.INTER_NEAREST)) 
    return    cv2.resize(cropped_image, (224,224),  interpolation=cv2.INTER_NEAREST)

def crop():
    def _crop(x):
        height, width = x.shape[:2]
        return  crop_center(x, int(width/2), int(width/2))
    return _crop

rescaler = 1

preprocess_function = preprocess_functions(p=0.8)
test_dir  = './dataset-a/train'
test_datagen = ImageDataGenerator(
                                   
                                   rescale=1./rescaler,
                                    preprocessing_function  = preprocess_functions(augmented=True),
                                #    horizontal_flip=True, vertical_flip=True,
                                #    horizontal_flip=True, vertical_flip=True,
                                #     validation_split=0.2,
                                #     width_shift_range=0.1,
                                #     height_shift_range=0.1,
                                #     fill_mode='wrap',
                                #     shear_range=0.1,
                                #     brightness_range=[0.95,1.05],
                                #     zoom_range=[0.95,1.05],
                                #     rotation_range=90
                                    
                                )

test_datagen_nopr = ImageDataGenerator(
                                    preprocessing_function  =preprocess_functions(),
                                   rescale=1./rescaler,
                                   validation_split=0.4/2,

                                )
                                
train_images = test_datagen.flow_from_directory(test_dir, target_size=(224,224),
                seed=100,
                batch_size=1,subset="training", class_mode="categorical", shuffle=True)
class_indices = train_images.class_indices

# Count the number of images per class
images_per_class = {}
for class_name, class_index in class_indices.items():
    images_per_class[class_name] = len(train_images.classes[train_images.classes == class_index])

print("Per Count for Training")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")
test_images = test_datagen.flow_from_directory(test_dir, target_size=(224,224),
                seed=10,
                batch_size=1,subset="validation", class_mode="categorical", shuffle=True)
test_images_nopr = test_datagen_nopr.flow_from_directory(test_dir, target_size=(224,224),
                seed=10,
                batch_size=32,subset="validation", class_mode="categorical", shuffle=True)

class_indices = test_images_nopr.class_indices

# Count the number of images per class
images_per_class = {}
for class_name, class_index in class_indices.items():
    images_per_class[class_name] = len(test_images_nopr.classes[test_images_nopr.classes == class_index])

print("Per Count for Validation")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")
# plt.figure(figsize=(10, 10))

# Plot images from the first generator
# for i, (images, _) in enumerate(test_images):
#     if i >= 1:
#         break
#     f = images[10]
#     ax = plt.subplot(1, 2, 1)
#     plt.imshow(geodesic_reconstruction_MMCE(f,2, 2).astype("uint8"))
#     plt.title('Mutli hat transformed')
#     f = images[5]
#     ax = plt.subplot(1, 2, 2)
#     plt.imshow(f.astype("uint8"))
#     plt.title('Original Image')

# plt.show()
class_names = list(test_images.class_indices.keys())
plt.figure(figsize=(10, 10))
for images, labels in  next(zip(test_images)):
        for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow((images[i]*rescaler).astype("uint8"))
                plt.title(class_names[np.argmax(labels[i])])
                plt.axis("off")
plt.figure(figsize=(10, 10))
for images, labels in  next(zip(test_images_nopr)):
        for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow((images[i]*rescaler).astype("uint8"))
                plt.title(class_names[np.argmax(labels[i])])
                plt.axis("off")
plt.show()