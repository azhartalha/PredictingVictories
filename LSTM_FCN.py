import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import matplotlib
import pickle
import sklearn

from keras.layers import Embedding, Dense, LSTM
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

# Model configuration
batch_size = 128
loss_function = BinaryCrossentropy()
max_sequence_length = 521
number_of_epochs = 20
#optimizer = Adam(learning_rate = 0.001)
optimizer = Adam()
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

# Reference for the model: https://github.com/titu1994/MLSTM-FCN/blob/master/acvitivity_model.py

xx = Masking()(input_layer)
xx = LSTM(8)(xx)
xx = Dropout(0.8)(xx)

yy = Permute((2, 1))(input_layer)
yy = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(yy)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)
yy = squeeze_excite_block(yy)

yy = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(yy)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)
yy = squeeze_excite_block(yy)

yy = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(yy)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)

yy = GlobalAveragePooling1D()(yy)

xx = concatenate([xx, yy])

out = Dense(1, activation='sigmoid')(xx)

model = Model(input_layer, out)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.keras.metrics.AUC()])
        
# Give the model summary
model.summary()

# Train the model
history = model.fit(x, y, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)



