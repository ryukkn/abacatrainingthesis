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
import splitfolders

import os
from random import sample
import shutil

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
train_dir = './dataset/train'
augmented_dir = './dataset/augmented'
val_dir = './dataset/val'
test_dir = './dataset/test'
EPOCHS = 20
NUM_CLASSES = 8 
IMAGE_SIZE = 224

models = ["Small", "Large"]
splits = [0.8, 0.7, 0.6]
balances = [False, True]
batches = [16,32,64]

def balance_directory(root_dir):
    # Get list of subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Find the subdirectory with the maximum number of files
    max_files = 999999999999999999999
    max_dir = ''
    perclass ={}
    for subdir in subdirs:
        num_files = len(os.listdir(os.path.join(root_dir, subdir)))
        perclass[subdir] = num_files
        if num_files < max_files:
            max_files = num_files
            max_dir = subdir

    # Remove excess files from the directory with the maximum number of files
    for subdir in subdirs:
        if subdir == max_dir:
            continue
        source_dir = os.path.join(root_dir, subdir)
        files_to_remove = os.listdir(source_dir)[max_files:]
        for file_to_remove in files_to_remove:
            os.remove(os.path.join(source_dir, file_to_remove))
            continue



def split_data(split):
    # remove existing split
    # shutil.rmtree("./dataset")
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio("../../../../nodeserver/data/grades", output="dataset",
        seed=1337, ratio=(split, (1-split)/2, (1-split)/2),group_prefix=None, move=False) # default values

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
        return _preprocess_functions
    shutil.copytree( './dataset/train',  './dataset/augmented')
    # Define your image directory
    image_dir = './dataset/augmented'

    # Get list of all folders in the directory
    folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]

    # Initialize an empty list to store image files
    image_files = []

    # Iterate through each folder
    for folder in folders:
        # Get the path to the current folder
        folder_path = os.path.join(image_dir, folder)
        
        # Get list of all image files in the current folder
        folder_image_files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Extend the image_files list with the image files from the current folder
        image_files.extend(folder_image_files)


    # Apply preprocessing to each image
    for image_file in image_files:
        print(image_file)
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (224, 224))
        # Apply preprocessing function
        preprocessed_image = preprocess_functions(innerMove=True)(resized_image)
        
        # Save the preprocessed image
        cv2.imwrite(image_file, preprocessed_image)

    print("Preprocessing complete.")

last_split = 0.7
skip = 16
i = 0
for split in splits:
    print(split)
    if split != last_split:
        split_data(split=split)
        last_split = split
    for balance in balances:
        if balance and i > skip:
            print("Balanced")
            balance_directory("./dataset/train")
            balance_directory("./dataset/val")
            balance_directory("./dataset/augmented")
            balance_directory("./dataset/test")
        else:
            print("Unbalanced")
        for model in models:
            print(model)
            if model == "Small":
                pretrained_model = tf.keras.applications.MobileNetV3Small(
                                    input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                    weights="imagenet",
                                    include_top=False,
                                    # include_preprocessing=False
                                )
            else:
                pretrained_model = tf.keras.applications.MobileNetV3Large(
                                    input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                    weights="imagenet",
                                    include_top=False,
                                    # include_preprocessing=False
                                )
            for batch in batches:
                print(batch)
                EPOCHS = 20
                if i < skip:
                    i+=1
                    continue
                i+=1
                BATCH_SIZE = batch
                train_gen_unaugmented = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                            shuffle=True,
                                                                            batch_size=BATCH_SIZE,
                                                                            seed=20,
                                                                            label_mode = "categorical",
                                                                            image_size=(IMAGE_SIZE, IMAGE_SIZE))

                train_gen_augmented = tf.keras.utils.image_dataset_from_directory(augmented_dir,
                                                                            shuffle=True,
                                                                            batch_size=BATCH_SIZE,
                                                                            seed=20,
                                                                            label_mode = "categorical",
                                                                            image_size=(IMAGE_SIZE, IMAGE_SIZE))
                train_gen = train_gen_unaugmented.concatenate(train_gen_augmented)
                # train_gen = train_gen.shuffle(buffer_size=len(train_gen_unaugmented))

                val_gen = tf.keras.utils.image_dataset_from_directory(val_dir,
                                                                            shuffle=True,
                                                                            batch_size=BATCH_SIZE,
                                                                            seed=20,
                                                                            label_mode = "categorical",
                                                                            image_size=(IMAGE_SIZE, IMAGE_SIZE))

                test_gen = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                            shuffle=False,
                                                                            batch_size=BATCH_SIZE,
                                                                            seed=20,
                                                                            label_mode = "categorical",
                                                                            image_size=(IMAGE_SIZE, IMAGE_SIZE))


                AUTOTUNE = tf.data.AUTOTUNE
                train_gen = train_gen.prefetch(buffer_size=AUTOTUNE)
                val_gen = val_gen.prefetch(buffer_size=AUTOTUNE)
                test_gen = test_gen.prefetch(buffer_size=AUTOTUNE)
                print('Start model (Feature extraction method)...')
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip(),
                    tf.keras.layers.RandomBrightness(0.05),
                    tf.keras.layers.RandomZoom(0.05),
                    tf.keras.layers.RandomRotation(0.5),
                ])

                for layer in pretrained_model.layers:
                    pretrained_model.trainable = False

                inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
                x = data_augmentation(inputs)
                x = tf.keras.layers.Flatten()(pretrained_model.output)
                outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
                _model = tf.keras.Model(pretrained_model.input, outputs)
                _model.compile(optimizer=Adam(learning_rate=0.0001), 
                            loss='categorical_crossentropy', 
                            metrics=['accuracy'])
                
                _model.summary()



                earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
                if balance:
                    csv_logger = tf.keras.callbacks.CSVLogger(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str(' Balanced/train.log'))
                else:
                    csv_logger = tf.keras.callbacks.CSVLogger(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str('/train.log'))
                plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=1,
                                            min_lr=1e-8, factor=0.3, min_delta=0.01,
                                            verbose=1)

                if balance:
                    checkpointer = ModelCheckpoint(filepath=str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str(' Balanced/latest_checkpoint.h5'), verbose=1, save_best_only=True,
                                        monitor="val_accuracy", mode="max",
                                        )
                else:
                    checkpointer = ModelCheckpoint(filepath=str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str('/latest_checkpoint.h5'), verbose=1, save_best_only=True,
                                        monitor="val_accuracy", mode="max",
                                        )
                
     
                history = _model.fit(train_gen,
                                    epochs=EPOCHS,
                                    callbacks=[checkpointer,plateau, earlyStopping,csv_logger],
                                    validation_data=val_gen,
                                    )


                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']

                loss = history.history['loss']
                val_loss = history.history['val_loss']

                # Fine tune 
                print("Fine-tuning....")
                pretrained_model.trainable = True
                # Fine-tune from this layer onwards
                fine_tune_at = 150

                # # Freeze all the layers before the `fine_tune_at` layer
                for layer in pretrained_model.layers[:fine_tune_at]:
                    layer.trainable = False

                _model.compile(optimizer=Adam(learning_rate=1e-5), 
                            loss='categorical_crossentropy', 
                            metrics=['accuracy'])

                if balance:
                    csv_logger = tf.keras.callbacks.CSVLogger(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str(' Balanced/finetune.log'))
                else:
                    csv_logger = tf.keras.callbacks.CSVLogger(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str('/finetune.log'))
                

                history = _model.fit(train_gen,
                        epochs=len(acc) + 10,
                        callbacks=[checkpointer,plateau, earlyStopping,csv_logger],
                        validation_data=val_gen,
                    )



                acc += history.history['accuracy']
                val_acc += history.history['val_accuracy']

                loss += history.history['loss']
                val_loss += history.history['val_loss']

                epochs_range = range(len(acc))
                if balance:
                    _model = tf.keras.models.load_model(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str(' Balanced/latest_checkpoint.h5'))
                else:
                    _model = tf.keras.models.load_model(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str('/latest_checkpoint.h5'))
                scores = _model.evaluate(test_gen, verbose=1)
                print('Testing loss: ', scores[0])
                print('Testing accuracy: ', scores[1])
                
                # plt.figure(figsize=(8, 8))
                # plt.subplot(1, 2, 1)
                # plt.plot(epochs_range, acc, label='Training Accuracy')
                # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
                # plt.legend(loc='lower right')
                # plt.title('Training and Validation Accuracy')

                # plt.subplot(1, 2, 2)
                # plt.plot(epochs_range, loss, label='Training Loss')
                # plt.plot(epochs_range, val_loss, label='Validation Loss')
                # plt.legend(loc='upper right')
                # plt.title('Training and Validation Loss')
                # if balance:
                #     plt.savefig(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str(' Balanced/Performance.png'))
                # else:
                #     plt.savefig(str('./saved/best/')+str(model)+str('/Batch ')+str(batch)+str(' Split ') +str(split)+str('/Performance.png'))
                # plt.show() 
                # plt.close()