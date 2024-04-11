    # for layer in pretrained_model.layers:
    #     layer.trainable = True
    # last_output = pretrained_model.layers[-1].output
    # last_output = pretrained_model.output
    # x = GlobalMaxPooling2D()(last_output)
    # x = BatchNormalization()(x)
    # x = GlobalMaxPooling2D()(last_output)
    # x = BatchNormalization()(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(NUM_CLASSES, activation='softmax')(x)
    # model = Model(pretrained_model.input, x)

    # model = Sequential()
    # model.add(pretrained_model)
    # model.add(Dense(NUM_CLASSES, activation='softmax'))

import itertools
import math
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # needed for working with this dataset
from sklearn.feature_extraction.image import extract_patches_2d
from random import sample
from tensorflow.keras.utils import Sequence
# import tensorflow_addons as tfa
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)
# Get the GPU memory fraction to allocate
# gpu_memory_fraction = 1

# Create GPUOptions with the fraction of GPU memory to allocate
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

# # Create a session with the GPUOptions
# session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

dataset_dir = '../../../../nodeserver/data/grades'

EPOCHS = 30
NUM_CLASSES = 8 
BATCH_SIZE = 16
split =0.2
IMAGE_SIZE = 224

def geodesic_reconstruction_mmce(f, B, n):
    # Initialization
    fE = f.copy()

    for i in range(n):
        # Calculation of top-hat transform by reconstruction
        gammaR_Bi = cv2.morphologyEx(f, cv2.MORPH_BLACKHAT, B)
        f_minus_gammaR_Bi = f - gammaR_Bi
        phiR_Bi = cv2.morphologyEx(B, cv2.MORPH_TOPHAT, f)
        RBTHi = phiR_Bi - f
        RWTHi = f_minus_gammaR_Bi

        # Calculation of subtractions from neighboring scales
        if i > 0:
            RWTHSi_minus_1 = RWTHi - RWTHi_minus_1
            RBTHSi_minus_1 = RBTHi - RBTHi_minus_1

        # Update variables for the next iteration
        RWTHi_minus_1 = RWTHi
        RBTHi_minus_1 = RBTHi

    # Maximum values of all the multiple scales obtained
    fCw = np.max(RWTHi)
    fCb = np.max(RBTHi)
    fDw = np.max(RWTHSi_minus_1)
    fDb = np.max(RBTHSi_minus_1)

    # Medical images contrast enhancement calculation
    fE = f + (fCw + fDw) - (fCb + fDb)

    return fE


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
    # threshold = 100  # Adjust as needed
    # x[np.sum(x < threshold, axis=2) == 3] = [255, 255, 255]
    return x.astype(np.float64)
# , 96  128, 156




class MergedGenerator(Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2
        self.filenames = generator1.filenames + generator2.filenames
        self.class_indices = {**generator1.class_indices, **generator2.class_indices}
        self.classes = list(self.class_indices.keys())
    def __len__(self):
        return len(self.generator1) + len(self.generator2)

    def __getitem__(self, index):
        if index < len(self.generator1):
            return self.generator1[index]
        else:
            return self.generator2[index - len(self.generator1)]
 


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
    
    return  foreground_extractor(cv2.resize(cropped_image, (IMAGE_SIZE,IMAGE_SIZE),  interpolation=cv2.INTER_NEAREST)) 


    

def preprocess_functions(p=1.0, size_list=[64, 72, 96], innerMove=False):
    def _preprocess_functions(x):
        height, width = x.shape[:2]
        # Inner move
        if(np.random.random() < p) and innerMove:
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
        return x
        # return (x/127.5)-1
    return _preprocess_functions


train_datagen_keras_processing = ImageDataGenerator(
                                    preprocessing_function = preprocess_functions(innerMove=True),
                                    horizontal_flip=True, vertical_flip=True,
                                    validation_split=split,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    fill_mode='wrap',
                                    shear_range=0.1,
                                    brightness_range=[0.95,1.05],
                                    zoom_range=[0.95,1.05],
                                    rotation_range=90
                                )


train_datagen_unaugmented = ImageDataGenerator(
                                    validation_split=split,
                                    preprocessing_function = preprocess_functions(),
                                   )
test_datagen = ImageDataGenerator(validation_split=split/2, 
                                  preprocessing_function = preprocess_functions(),
                                  )
    

# train_gen_innermove = train_datagen_innermove.flow_from_directory(dataset_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
#                                               batch_size=BATCH_SIZE,subset="training" , class_mode="sparse", shuffle=True)
# train_gen_keras_processing = train_datagen_keras_processing.flow_from_directory(dataset_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
#                                               batch_size=BATCH_SIZE,subset="training" , class_mode="sparse", shuffle=True, seed=30)

train_gen_augmented = train_datagen_keras_processing.flow_from_directory(dataset_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size=BATCH_SIZE,subset="training" , class_mode="sparse", shuffle=True, seed=30)


train_gen_unaugmented = train_datagen_unaugmented.flow_from_directory(dataset_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size=BATCH_SIZE,subset="training" , class_mode="sparse", shuffle=True, seed=30)



class_indices = train_gen_unaugmented.class_indices

# Count the number of images per class
images_per_class = {}
for class_name, class_index in class_indices.items():
    images_per_class[class_name] = len(train_gen_unaugmented.classes[train_gen_unaugmented.classes == class_index])

print("Per Count for Training (Not Augmented)")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")


# train_gen_augmented = MergedGenerator(train_gen_innermove, train_gen_keras_processing)
train_gen = MergedGenerator(train_gen_augmented, train_gen_unaugmented)
# train_gen = train_gen_augmented
# train_gen = train_gen_augmented

class_indices = train_gen.class_indices

for class_name, class_index in class_indices.items():
    images_per_class[class_name] += len(train_gen_augmented.classes[train_gen_augmented.classes == class_index])

print("Per Count for Training (Augmented)")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")

val_gen = test_datagen.flow_from_directory(dataset_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size=BATCH_SIZE,subset="validation",class_mode="sparse")


class_indices = train_gen.class_indices

# Count the number of images per class
images_per_class = {}
for class_name, class_index in class_indices.items():
    images_per_class[class_name] = len(val_gen.classes[val_gen.classes == class_index])

print("Per Count for Validation")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")

test_gen = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size=BATCH_SIZE,subset="validation",class_mode="sparse", seed=1200)


# Count the number of images per class
images_per_class = {}
for class_name, class_index in class_indices.items():
    images_per_class[class_name] = len(test_gen.classes[test_gen.classes == class_index])

print("Per Count for Testing")
# Print the number of images per class
for class_name, count in images_per_class.items():
    print(f"Class: {class_name}, Number of Images: {count}")


pretrained_model = tf.keras.applications.MobileNetV3Small(
                        input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                        weights="imagenet",
                        include_top=False,
                        # include_preprocessing=False
                    )
try:
    model = tf.keras.models.load_model('./saved/latest_checkpoint.h5')
    model.summary()
    
    pretrained_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    print('Fine tuning model...')
    EPOCHS = 10
except:
    print('Start model (Feature extraction method)...')
    for layer in pretrained_model.layers:
        pretrained_model.trainable = False
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    # x = tf.keras.layers.Dense(576, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(pretrained_model.input, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    
    model.summary()



earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

# val_acc with min_delta 0.003; val_loss with min_delta 0.01
plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3,
                            min_lr=1e-8, factor=0.3, min_delta=0.01,
                            verbose=1)
checkpointer = ModelCheckpoint(filepath='./saved/latest_checkpoint.h5', verbose=1, save_best_only=True,
                               monitor="val_accuracy", mode="max",
                              )


total_samples = len(train_gen.filenames)

history = model.fit(train_gen,
                    epochs=EPOCHS,
                    callbacks=[checkpointer,plateau, earlyStopping],
                    validation_data=val_gen,
                    )




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

if len(acc) < EPOCHS:
    print("\nStopped with stopping function\n")

# Fine tune 
print("\nFine-tuning.....\n")
pretrained_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 150

# # Freeze all the layers before the `fine_tune_at` layer
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

history_finetuned = model.fit(train_gen,
                    epochs=len(acc) + 10,
                    callbacks=[checkpointer,plateau, earlyStopping],
                    validation_data=val_gen,
                    )


scores = model.evaluate(test_gen, verbose=1)
print('Testing loss (Fine tuned): ', scores[0])
print('Testing accuracy (Fine tuned): ', scores[1])

acca = history.history['accuracy']
val_acca = history.history['val_accuracy']

lossa = history.history['loss']
val_lossa = history.history['val_loss']

scoresa = model.evaluate(test_gen, verbose=1)
print('Testing loss: ', scoresa[0])
print('Testing accuracy: ', scoresa[1])

epochs_range = range(len(acca))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acca, label='Training Accuracy')
plt.plot(epochs_range, val_acca, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, lossa, label='Training Loss')
plt.plot(epochs_range, val_lossa, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()