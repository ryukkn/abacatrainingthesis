
import zipfile

from tensorflow.keras.applications import MobileNetV3Small as MobileNetV3
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping)
import numpy as np
import matplotlib.pyplot as plt
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

EPOCHS = 150

INPUT_SIZE = 224

# with tf.device(tf.DeviceSpec(device_type="GPU")):
if (1):

        # Set train data generator with data augmentation.
        # train_image_gen = ImageDataGenerator(
        #         # featurewise_center=False,
        #         # samplewise_center=False,
        #         # featurewise_std_normalization=False,
        #         # samplewise_std_normalization=False,
        #         # rotation_range=40,
        #         # width_shift_range=0.1,
        #         # height_shift_range=0.1,
        #         # shear_range=0.2,
        #         # zoom_range=0.2,
        #         # channel_shift_range=0.2,
        #         # fill_mode='nearest',
        #         # horizontal_flip=True,
        #         # vertical_flip=True,
        #         rescale=1./255,
        #         preprocessing_function=None,
        #         data_format='channels_last',
        #         validation_split=0.1
        # )

        # Set validation data generator.
        # validation_image_gen = ImageDataGenerator(
        #         rescale=1./255,
        #         validation_split=0.1)

        data_dir = '../../../../nodeserver/data/grades/'

        # traingen = train_image_gen.flow_from_directory(
        #                 '../../../../nodeserver/data/grades/',
        #                 target_size=(INPUT_SIZE, INPUT_SIZE),
        #                 batch_size=BATCH_SIZE,
        #                 subset='training'
        #                 # class_mode='binary'
        #                 )
        train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(INPUT_SIZE, INPUT_SIZE),
                batch_size=BATCH_SIZE)

        # Data generator for validation.
        # validgen = validation_image_gen.flow_from_directory(
        #                 '../../../../nodeserver/data/grades/',
        #                 target_size=(INPUT_SIZE, INPUT_SIZE),
        #                 batch_size=BATCH_SIZE,
        #                 subset='validation'
        #                 # class_mode='binary'
        #                 )

        val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(INPUT_SIZE, INPUT_SIZE),
                batch_size=BATCH_SIZE)

        # class_names = train_ds.class_ind
        class_names = train_ds.class_names
        print(class_names)

        NUM_CLASSES = len(class_names)
        print(NUM_CLASSES)
        
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
                for i in range(9):
                        ax = plt.subplot(3, 3, i + 1)
                        plt.imshow(images[i].numpy().astype("uint8"))
                        plt.title(class_names[labels[i]])
                        plt.axis("off")
        plt.show()
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
        # Loading pretrained model on imagenet with global average pooling and
        # original sized convolutional filters.
        try:
                model = tf.keras.models.load_model('../MobilenetV3/saved/mobilenetv3TF.h5')
                print('Continue model..')
        except:
                print('Creating new model')
                pretrained_model = MobileNetV3(input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                                        # alpha=1.0,
                                        include_top=False,
                                        # classes=NUM_CLASSES,
                                        weights='imagenet',
                                        # pooling='avg'
                                        )
                preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
                # Model topology.
                base = pretrained_model
                base.trainable = False
                base.summary()
                image_batch, label_batch = next(iter(train_ds))
                feature_batch = base(image_batch)
                global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                feature_batch_average = global_average_layer(feature_batch)
                prediction_layer = tf.keras.layers.Dense(NUM_CLASSES)
                prediction_batch = prediction_layer(feature_batch_average)

                # model = Sequential()
                # model.add(base) 
                # model.add(Dense(NUM_CLASSES))
                # model.add(Activation('sigmoid'))
                # model.summary()

                inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
                # x = data_augmentation(inputs)
                x = preprocess_input(inputs)
                x = base(x, training=False)
                x = global_average_layer(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = prediction_layer(x)
                model = tf.keras.Model(inputs, outputs)
                model.summary()

                len(model.trainable_variables)
                # Optimizer settings.
                # opt = RMSprop(lr=0.005, decay=1e-6)
                # opt = RMSprop(lr=0.01, decay=1e-6)

                
                
                # model.compile(
                #         optimizer='adam',
                #         # optimizer=opt, 
                #         #       loss='binary_crossentropy', 
                #               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                #               metrics=['accuracy'])

        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        # Extract dataset from the repository zipped folder.
        # zip_object = zipfile.ZipFile('../../../data/raw/cats_and_dogs_filtered.zip')
        # zip_object.extractall('../../../data/raw/')
        # zip_object.close()

        # Data generator for training.

        # Callbacks for training.
        # rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
        # es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, mode='auto')
        # mc = ModelCheckpoint('./saved/checkpoints/', monitor='val_accuracy',
        #                 save_best_only=True)

        # Fit the model using the generators and the callbacks above defined.

        history = model.fit(train_ds,
                #   steps_per_epoch=len(traingen),
                    epochs=EPOCHS,
                #     shuffle=True,
                #     callbacks=[rlr, es, mc],
                # validation_steps=len(validgen) ,
                    validation_data=val_ds)
        


# Trained model's score.
scores = model.evaluate(val_ds, verbose=1)
print('Validation loss: ', scores[0])
print('Validation accuracy: ', scores[1])



model.save('saved/mobilenetv3TF.h5')
 
import tensorflow as tf

model = tf.keras.models.load_model('saved/mobilenetv3TF.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("saved/model.tflite", "wb").write(tflite_model)

# url = "../../../../nodeserver/data/grades/EF/EF-0005.jpg"
# ef_path = tf.keras.utils.get_file('EF', origin=url)

# img = tf.keras.utils.load_img(
#     ef_path, target_size=(INPUT_SIZE, INPUT_SIZE)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
