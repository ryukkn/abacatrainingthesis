import tensorflow as tf

model = tf.keras.models.load_model('./saved/best/Large/Batch 16 Split 0.8/latest_checkpoint.h5')
# model = tf.keras.models.load_model('./saved/latest_checkpoint.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("saved/model.tflite", "wb").write(tflite_model)
