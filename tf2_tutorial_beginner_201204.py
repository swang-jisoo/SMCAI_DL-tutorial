#####
# MNIST handwritten digit recognition
# MNIST dataset is a collection of 60,000 small square 28*28 pixel grayscale images of handwritten single digits
# between 0-9 inclusive. The objective of this project is to classify the given image of a handwritten digit
# into one of 10 classes representing integer values from 0-9.

# ref:
# https://www.tensorflow.org/tutorials/quickstart/beginner
#####

# Beginner ver.
# Import necessary libraries
import tensorflow as tf

'''# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # >0.3 doesn't work with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)'''

# Load mnist handwritten dataset
mnist = tf.keras.datasets.mnist

# Split dataset into training & test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data for faster learning and better performance by reducing variance
# Convert the pixel range from 0(black)-255(white) to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a model
# Sequential(): a model for a plain stack of layers where each layer has one input and one output tensors.
# Flatten(): to fully connect all pixels to hidden layer, reshape two dimensional arrays (28, 28)
# into one dimensional array (28*28, 1)
# Dense(units): create a densely connected layer, generating a unit dimensionality (node, shape) of
# output (= activation(dot(input, kernel) + bias))
# Relu (Rectified Linear) activation: max(0,x)
# Dropout: randomly drop out the 20% of node (and corresponding edges)
# --> reduced variance, preventing over-fitting (NOTE: training only)
# Dense(units): extract the 10 node values (= output class)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Adam optimization: SGD with momentum (moving average of the gradient) + RMSProp (squared gradients to scale
# the learning rate)
# Sparse categorical cross entropy loss function:
# - cross entropy: uncertainty where the info can be incorrect; compare the probability and the actual value
# (if prob == actual, cross entropy = entropy; else cross entropy > entropy)
# - sparse categorical cross entropy: with large number of classes, convert labels into one hot embedding of labels
# to speed up the execution
# Accuracy: the proportion of true results among the total number of cases examined
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
# Epoch: running the entire dataset
model.fit(x_train, y_train, epochs=5)

# Test the model with test set
model.evaluate(x_test, y_test, verbose=2)
