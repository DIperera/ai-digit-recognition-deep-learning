from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf

# Load trained model from Drive
model = tf.keras.models.load_model('/content/drive/MyDrive/mnist_model.h5')