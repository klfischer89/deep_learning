import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(1, name = "neuron", activation = "sigmoid")
])

model.compile(
    optimizer = keras.optimizers.RMSprop(0.01),
    loss = keras.losses.BinaryCrossentropy()
)

model.fit(X.astype(np.float32), y, batch_size = 64, epochs = 100)

