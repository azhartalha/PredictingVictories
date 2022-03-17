from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf

x = []
y = []

with open('x.pkl', 'rb') as f:
    x = pickle.load(f)

with open('y.pkl', 'rb') as f:
    y = pickle.load(f)

x = tf.keras.preprocessing.sequence.pad_sequences(x, padding="post", value=99.0, dtype='float32')
y = np.array(y, dtype='float32')

features = 40
time_steps = 521 # Max time steps

model = Sequential()
model.add(tf.keras.layers.Masking(mask_value=99.0, input_shape=(time_steps, features)))
model.add(tf.keras.layers.GRU(8, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=[tf.keras.metrics.AUC()])

history = model.fit(x, y, validation_split=0.2, epochs=50, batch_size=128)

