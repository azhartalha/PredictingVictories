from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

features = 8 # Health points of 8 units
time_steps = 119 # Max time steps

model = Sequential()
model.add(tf.keras.layers.Masking(mask_value=99.0, input_shape=(time_steps, features)))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, validation_split=0.2, epochs=50, batch_size=32)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["training_acc", "validation_acc"])
plt.show()

