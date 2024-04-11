
import zipfile

from tensorflow.keras.applications import MobileNetV3Small as MobileNetV3
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping)
import keras

import os

import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6072)])


# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} ) 
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.clear_session() 
# tf.compat.v1.keras.backend.set_session(sess)


# sess.run
# tf.compat.v1.keras.backend.set_session(sess)

# gpu_memory_fraction = 1

# Create GPUOptions with the fraction of GPU memory to allocate
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)

# Create a session with the GPUOptions
# session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,device_count = {'GPU': 1 , 'CPU': 6} ))
# tf.compat.v1.keras.backend.set_session(session)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
# logical_gpus = tf.config.list_logical_devices('GPU')
# print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
BATCH_SIZE = 32

EPOCHS = 50

INPUT_SIZE = 224

if(1):

        # Set train data generator with data augmentation.
        train_image_gen = ImageDataGenerator(
                # featurewise_center=False,
                # samplewise_center=False,
                # featurewise_std_normalization=False,
                # samplewise_std_normalization=False,
                # rotation_range=40,
                # width_shift_range=0.1,
                # height_shift_range=0.1,
                # shear_range=0.2,
                # zoom_range=0.2,
                # channel_shift_range=0.2,
                # fill_mode='nearest',
                # horizontal_flip=True,
                # vertical_flip=True,
                rescale=1./255,
                # preprocessing_function=None,
                # data_format='channels_last',
                validation_split=0.0
        )

        # Set validation data generator.
        validation_image_gen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.1)

        traingen = train_image_gen.flow_from_directory(
                        '../../../../nodeserver/data/grades/',
                        target_size=(INPUT_SIZE, INPUT_SIZE),
                        batch_size=BATCH_SIZE,
                        subset='training'
                        # class_mode='binary'
                        )

        # Data generator for validation.
        validgen = validation_image_gen.flow_from_directory(
                        '../../../../nodeserver/data/grades/',
                        target_size=(INPUT_SIZE, INPUT_SIZE),
                        batch_size=BATCH_SIZE,
                        subset='validation'
                        # class_mode='binary'
                        )

        class_names = traingen.class_indices

        print(class_names)

        NUM_CLASSES = len(class_names)

        # Loading pretrained model on imagenet with global average pooling and
        # original sized convolutional filters.
        try:
                model = tf.keras.models.load_model('../MobilenetV3/saved/mobilenetv3TF.h5')
                print('Continue model..')
        except:
                print('Creating new model')
                pretrained_model = MobileNetV3(input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                                        alpha=1.0,
                                        include_top=False,
                                        classes=NUM_CLASSES,
                                        weights='imagenet',
                                        pooling='avg')

                # Model topology.
                base = pretrained_model
                base.trainable = False
                model = Sequential()
                model.add(base) 
                model.add(Dense(NUM_CLASSES))
                model.add(Activation('sigmoid'))

                model.summary()
                # Optimizer settings.
                # opt = RMSprop(lr=0.005, decay=1e-6)
                opt = RMSprop(lr=0.01, decay=1e-6)
                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])



        # Extract dataset from the repository zipped folder.
        # zip_object = zipfile.ZipFile('../../../data/raw/cats_and_dogs_filtered.zip')
        # zip_object.extractall('../../../data/raw/')
        # zip_object.close()

        # Data generator for training.

        # Callbacks for training.
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
        es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, mode='auto')
        # mc = ModelCheckpoint('../MobilenetV3/saved/checkpoint.hd5', monitor='val_loss',
        #                 save_best_only=True)

        # Fit the model using the generators and the callbacks above defined.

        model.fit(traingen,
                  steps_per_epoch=len(traingen),
                    epochs=EPOCHS,
                    shuffle=True,
                    callbacks=[rlr, es],
                validation_steps=len(validgen) ,
                    validation_data=validgen)

# Trained model's score.
scores = model.evaluate(validgen, verbose=1)
print('Validation loss: ', scores[0])
print('Validation accuracy: ', scores[1])

model.save('saved/mobilenetv3TF.h5')
 