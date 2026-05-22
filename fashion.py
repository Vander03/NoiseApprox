import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# This automatically downloads the data into NumPy arrays
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Save them as .npy files
np.save('trainX.npy', x_train)
np.save('trainY.npy', y_train)
np.save('testX.npy', x_test)
np.save('testY.npy', y_test)

print(x_train.shape)