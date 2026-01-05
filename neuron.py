import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers

df = pd.read_csv("./data/diamonds.csv.bz2")

X = df[["carat"]]
y = df["price"]

model = keras.Sequential([
    layers.Dense(1, name = "neuron")
])

v = tf.Variable([
    [1],
    [3]
])

model.compile(
    optimizer = keras.optimizers.RMSprop(),
    loss = keras.losses.MeanSquaredError()
)

model.fit(X.astype(np.float32), y, batch_size = 1, epochs = 1)

model.predict(np.array([
    [0.1]
]))