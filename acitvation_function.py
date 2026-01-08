import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age"]]
y = df["Outcome"]

model = keras.Sequential([
    layers.Dense(1, name = "neuron", activation = "sigmoid")
])

model.compile(
    optimizer = keras.optimizers.RMSprop(0.01),
    loss = keras.losses.BinaryCrossentropy()
)

model.fit(X.astype(np.float32), y, batch_size = 64, epochs = 100)

np.mean((model.predict(X) > 0.5).ravel() == y)