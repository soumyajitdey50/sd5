import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model2.h5')
converter = tf.lite.TFLiteConverter.from_saved_model(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)