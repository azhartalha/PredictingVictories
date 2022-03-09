import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import matplotlib
import pickle
import sklearn

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

# Model configuration
additional_metrics = ['accuracy']
batch_size = 16
loss_function = BinaryCrossentropy()
max_sequence_length = 119
number_of_epochs = 1500
optimizer = Adam(learning_rate = 0.001)
validation_split = 0.20
verbosity_mode = 1

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load dataset
x = []
y = []

# Need dataset with same length. Cannot use Keras masking and padding.
with open('x_same_length.pkl', 'rb') as f:
    x = pickle.load(f)
x = np.array(x)
    
with open('y.pkl', 'rb') as f:
    y = pickle.load(f)

x = np.array(x)
y = np.array(y, dtype='float32')

input_shape = x.shape[1:]

input_layer = keras.layers.Input(input_shape)

# Reference for the model: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
# From the following paper: H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, 
# "Deep learning for time series classification: a review," Data Mining and Knowledge Discovery, 
# vol. 33, no. 4, Springer US, 2019, pp. 917–63, doi: 10.1007/s10618-019-00619-1.

# ********* Slightly modified to accomodate for binary classification *********
# ********* NEED TO CONFIRM THAT THE CHANGES ARE CORRECT **********************

conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation(activation='relu')(conv1)

conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

# FINAL

gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)
        
# Give the model summary
model.summary()

# Train the model
history = model.fit(x, y, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)



