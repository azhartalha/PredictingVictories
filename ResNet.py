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

n_feature_maps = 64

nb_classes = 2

input_shape = x.shape[1:]

input_layer = keras.layers.Input(input_shape)

# Reference for the model: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
# From the following paper: H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, 
# "Deep learning for time series classification: a review," Data Mining and Knowledge Discovery, 
# vol. 33, no. 4, Springer US, 2019, pp. 917–63, doi: 10.1007/s10618-019-00619-1.

# ********* Slightly modified to accomodate for binary classification *********
# ********* NEED TO CONFIRM THAT THE CHANGES ARE CORRECT **********************

 # BLOCK 1

conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
conv_x = keras.layers.BatchNormalization()(conv_x)
conv_x = keras.layers.Activation('relu')(conv_x)

conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
conv_y = keras.layers.BatchNormalization()(conv_y)
conv_y = keras.layers.Activation('relu')(conv_y)

conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
conv_z = keras.layers.BatchNormalization()(conv_z)

# expand channels for the sum
shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

output_block_1 = keras.layers.add([shortcut_y, conv_z])
output_block_1 = keras.layers.Activation('relu')(output_block_1)

# BLOCK 2

conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
conv_x = keras.layers.BatchNormalization()(conv_x)
conv_x = keras.layers.Activation('relu')(conv_x)

conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
conv_y = keras.layers.BatchNormalization()(conv_y)
conv_y = keras.layers.Activation('relu')(conv_y)

conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
conv_z = keras.layers.BatchNormalization()(conv_z)

# expand channels for the sum
shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

output_block_2 = keras.layers.add([shortcut_y, conv_z])
output_block_2 = keras.layers.Activation('relu')(output_block_2)

# BLOCK 3

conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
conv_x = keras.layers.BatchNormalization()(conv_x)
conv_x = keras.layers.Activation('relu')(conv_x)

conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
conv_y = keras.layers.BatchNormalization()(conv_y)
conv_y = keras.layers.Activation('relu')(conv_y)

conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
conv_z = keras.layers.BatchNormalization()(conv_z)

# no need to expand channels because they are equal
shortcut_y = keras.layers.BatchNormalization()(output_block_2)

output_block_3 = keras.layers.add([shortcut_y, conv_z])
output_block_3 = keras.layers.Activation('relu')(output_block_3)

# FINAL

gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)
        
# Give the model summary
model.summary()

# Train the model
history = model.fit(x, y, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)


