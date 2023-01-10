import tensorflow as tf
from keras.models import load_model
model = load_model("/content/drive/MyDrive/deeppicar/output/lane_navigation_check.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("/content/drive/MyDrive/deeppicar/output/lane_navigation_check.tflite", "wb") .write(tfmodel)