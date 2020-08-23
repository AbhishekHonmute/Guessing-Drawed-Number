import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.00, x_test / 255.00

model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(256, activation = 'relu'),
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dense(10)
	])

model.compile(
	optimizer = 'adam',
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics = ['accuracy']
	)

model.fit(x_train, y_train, epochs = 10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)

print("\n\nTest loss : " + str(test_loss) + "\nTest accurecy : " + str(test_acc * 100))

model.save("./mnist_model_file_4/mnist_model")
# 0 : 97.89
# 1 : 98.07
# 2 : 97.93
# 3 : 98.24