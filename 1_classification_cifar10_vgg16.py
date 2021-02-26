#####
# Image Classification

# dataset: CIFAR-10
# description: a labeled set of colored images
# images: shape=(32, 32, 3)
# labels: num_classes=10 (0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, and 9=truck)
# total_num_examples=60000
# split: training=50000 (5000 images per class), test=10000 (1000 randomly-selected images per class)
# ref: https://www.cs.toronto.edu/~kriz/cifar.html

# model: VGG16 (Deep Convolutional Neural Network - DCNN)
# ref. paper: (VGG16 paper) VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
# idea: increase the depth of networks by using a small (3x3) convolution filters in all layers
# 3x3 convolution + maxpooling
# contribution: outperforming results using CNN with substantially increased depth

# lang: Tensorflow v2, functional API
# ref: https://www.tensorflow.org/guide/keras/functional
#####


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Set the value of hyper-parameters
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
CLASSES = 10
# UPSAMPLING_SIZE = (2,2)
EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
VERBOSE = 1

# Results by hyper-parameters
# ==> LEARNING_RATE=0.001 (zigzag result of training acc)
# ==> EPOCHS=20; LEARNING_RATE=0.0001; BATCH_SIZE=16; loss: 1.1909 - accuracy: 0.7927
# ==> EPOCHS=20; LEARNING_RATE=0.0001; BATCH_SIZE=32; loss: 1.0698 - accuracy: 0.7885
# ==> EPOCHS=20; LEARNING_RATE=0.00005; BATCH_SIZE=32; loss: 1.2411 - accuracy: 0.7474
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=32; loss: 1.2150 - accuracy: 0.7998
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=32; UPSAMPLING_SIZE=(2,2); loss: 0.9695 - accuracy: 0.8239


# Load the dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the dataset
# Data-type conversion: uint8 --> float32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Normalization: pixel range = 0(black)-255(white) --> 0-1
# advantage:
# - theoretically it should not affect NN's performance; but empirical evidences show that it improves performance and training time
x_train, x_test = x_train / 255.0, x_test / 255.0
# One-hot encoding: label encoding = [6] --> one-hot encoding = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# advantage:
# - eliminate the possibility that categorical features are misinterpreted as having a hierarchical relationship
# - improve the model performance
# disadvantage:
# - lead to high memory consumption if num of categories is large
y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, CLASSES)


# Build a VGG16 architecture
# input layer
# tf.keras.Input(shape, dtype)
# - shape: a shape of input tensor in a tuple of integers, excl. batch size
# by default (data_format='channels_last'), the ordering of the dimensions in 4D tensor is batch + (rows, cols, channels)
# - dtype: data type expected by the input, as a string (float32, float64, int32...)
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/Input
input_tensor = Input(shape=IMG_SHAPE, dtype='float32', name='input')
# up-sampling: if upsampling_size=(2,2), input_shape=(1,2,2,1) --> output_shape=(1,4,4,1)
# tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')
# - size: upsampling factors for rows & cols in int or tuple of 2 int
# - interpolation: one of 'nearest' or 'bilinear' in string
# nearest: [[1,2][3,4]]-->[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
# bilinear: [[1,2][3,4]]-->[[1,1.25,1.75,2],[1.5,1.75,2.25,2.5],[2.5,2.75,3.25,3.5],[3,3.25,3.75,4]]
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D
'''
upsampling = tf.keras.layers.UpSampling2D(size=UPSAMPLING_SIZE, 
                                          name='upsampling')(input_tensor)  # to check the performance improvement
'''

# CNN in classification = backbone (feature extractor) + classifier
# backbone (feature extractor)
# convolution layer
# : Capture connectivity by shifting a feature identifier(=receptive field=convolution window), called filter or kernel,
# by strides in each dimension over the image.
# - kernel: a 2D array of weights
# - filter: 3D structures of multiple kernels stacked together
# - stride: a length of shifts over the image matrix
# - advantage: local connectivity, weight sharing (compared to MLP)
# : As more convolutional layers are added, higher level of features based on the previous detected features will be detected.
# : (batch, rows, cols, channels) --> (batch, rows/strides, cols/strides, filters)
# tf.keras.layers.Conv2D(filters, kernel_size, strides=(1,1), padding='valid',
#                        activation=None, kernel_initializer='glorot_uniform')
# - filters: Integer, the dimensionality of the output space (the number of channels)
# - kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
# - strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
# - padding: one of 'valid' (no padding) or 'same' (zero padding)
# if stride = 1 & padding = 'valid', output_shape == (input_shape - kernel_size + 1) / strides
# if stride = 1 & padding = 'same', output_shape == input_shape
# if stride > 1, pad (kernel_size - 1) rows/cols of zeros evenly around the inputs such that
# output_shape == (input_shape + padding_factor) // strides
# ref: https://smartstuartkim.wordpress.com/2019/04/27/cnn-with-tensorflow-keras/
# - activation: activation(cov2d(inputs, kernel) + bias)
# some of widely used activation functions are:
# (1) sigmoid: 1 / (1 + exp(-x))
# (2) relu: max(x, 0)
# (3) leaky relu: max(0.1x, 0)
# (2) softmax: exp(x) / tf.reduce_sum(exp(x)); in range (0,1) & sum to 1; converts logit scores to probabilities
# built-in functions can be found at https://www.tensorflow.org/api_docs/python/tf/keras/activations
# - kernel_initializer: https://www.tensorflow.org/api_docs/python/tf/keras/initializers
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

# max pooling
# : down-samples the input representation by taking the maximum value over the window
# - advantage: highlight the most present feature in the window
# tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None(==pool_size), padding='valid')
# - pool_size: integer or tuple of 2 integers, window size over which to take the maximum
# - strides: Integer, tuple of 2 integers, or None. If None, it will default to pool_size.
# - padding: One of "valid" or "same"
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D

# block 1
conv1_1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1-1')(input_tensor)
conv1_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv1-2')(conv1_1)
maxpool1 = MaxPooling2D(2, padding='same', name='maxpool1')(conv1_2)  # down-sampling # 16,16,64
# block 2
conv2_1 = Conv2D(128, 3, activation='relu', padding='same', name='conv2-1')(maxpool1)
conv2_2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2-2')(conv2_1)
maxpool2 = MaxPooling2D(2, padding='same', name='maxpool2')(conv2_2)  # 8,8,128
# block 3
conv3_1 = Conv2D(256, 3, activation='relu', padding='same', name='conv3-1')(maxpool2)
conv3_2 = Conv2D(256, 3, activation='relu', padding='same', name='conv3-2')(conv3_1)
conv3_3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3-3')(conv3_2)
maxpool3 = MaxPooling2D(2, padding='same', name='maxpool3')(conv3_3)  # 4,4,256
# block 4
conv4_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-1')(maxpool3)
conv4_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-2')(conv4_1)
conv4_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-3')(conv4_2)
maxpool4 = MaxPooling2D(2, padding='same', name='maxpool4')(conv4_3)  # 2,2,512
# block 5
conv5_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-1')(maxpool4)
conv5_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-2')(conv5_1)
conv5_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-3')(conv5_2)
maxpool5 = MaxPooling2D(2, padding='same', name='maxpool5')(conv5_3)  # 1,1,512


# classifier
# flatten
# : reshapes the input into a 1d array of elements while preserving the batch size
# tf.keras.layers.Flatten()
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

# fully-connected (FC) layer
# : fully connects the input nodes to the output nodes
# - advantage: take each feature map as independent variable and pass them through activation function to classify them
# tf.keras.layers.Dense(units, activation=None)
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

# drop out
# : randomly sets input units to 0 with a frequency of rate at each step during training time
# - advantage: prevent overfitting by reducing the correlation between nodes
# tf.keras.layers.Dropout(rate)
# - rate: Float between 0 and 1. Fraction of the input units to drop.
# - ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

flatten = Flatten(name='flatten')(maxpool5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten) # unnecessary due to the final dimension size after block 5
# drop1 = Dropout(0.5)
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
# drop2 = Dropout(0.5)
fc3 = Dense(256, activation='relu', name='fc3')(flatten)  # NOTE: check input
output_tensor = Dense(CLASSES, activation='softmax', name='output')(fc3)

# Create a model
vgg16 = Model(input_tensor, output_tensor, name='vgg16')
vgg16.summary()  # plot the model architecture with the number of parameters (complexity)

# Compile/configure the model for training

# Forward propagation:
# with randomly initialized weight and bias, propagates to the hidden units at each layer and finally produce output.
# Backward propagation:
# goes back from the cost backward through the network in order to update weights
# use loss function and gradient optimizer to compute gradient and update weight

# Loss function:
# tells how good a model is at making prediction at a given set of parameters
#   - loss: the difference between model prediction and actual answer
# cross entropy: uncertainty where the info can be incorrect; compare the probability and the actual value
# (if pred == actual, cross entropy = entropy; else cross entropy > entropy)
# sparse categorical cross entropy: with large number of classes, convert labels into one hot embedding of labels

# Gradient Descent:
# Optimizes/minimizes the loss (cost) by stepping down (how the big the step is  = learning rate)
# over the cost function curve (from the top of mountain to the flat bottom of valley)
# in the direction of steepest descent (defined by the negative of the gradient (=slope)).
# Computes the cost gradient based on the full-batch (entire training dataset).
# --> slower to update te weights and longer to converge to the global cost minimum
# Stochastic gradient descent (SGD):
# update the weight after each mini-batch
# --> the path towards the global cost minimum may go zig-zag but surely faster to converge to the global cost minimum
# Adam optimization:
# SGD with momentum (moving average of the gradient) + RMSProp (squared gradients to scale the learning rate)

# Accuracy: the proportion of true results among the total number of cases examined

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)  # lower learning rate, better performance
vgg16.compile(loss='categorical_crossentropy',  # 'sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Fix the gpu memory issue
# Error: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Train the model
# tf.keras.Model.fit(x=None, y=None, batch_size=32, epochs=1,
#                    validation_split=0.0, validation_data=None, verbose=1, callbacks=None)
# - x: Input data
# - y: Target data
# - batch_size: Integer or None. Number of samples per gradient update.
# Do not specify the batch_size if your data is in the form of symbolic tensors, datasets, generators,
# or keras.utils.Sequence instances (since they generate batches).
# - epochs: Integer. a number of iterations over the entire x and y data provided to train the model
# - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data.
# - validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
# The model will not be trained on this data. validation_data will override validation_split.
# - verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
# - callbacks: utilities called at certain points during model training
# more info can be found at https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/callbacks
# - ref: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#fit
vgg16.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.2)

# Evaluate/predict the model with test set
# tf.keras.Model.evaluate(x, y): Returns the loss value & metrics values for the model in test mode.
# - ref: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#evaluate
# tf.keras.Model.predict(x): Generates output predictions for the input samples
# - ref: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#predict
# The both methods use the same interpretation rules, but as the explanation indicates, model.evaluate() uses for
# validation, while model.predict() uses for prediction.
vgg16.evaluate(x_test, y_test, verbose=VERBOSE)
prediction = vgg16.predict(x_test)
print(prediction)
