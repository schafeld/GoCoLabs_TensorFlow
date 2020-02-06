import tensorflow as tf
import numpy as np
from tensorflow import keras

# define and compile neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

# provide data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train model
model.fit(xs, ys, epochs=500)

# prediction for x = 10
# we KNOW by looking at the numbers for x and y the equation has to be Y=3X+1
# simply by looking at the Numpy arrays above, so Y = 31 for X = 10.
# Now what will the AI guess/predict:
print(model.predict([10.0]))