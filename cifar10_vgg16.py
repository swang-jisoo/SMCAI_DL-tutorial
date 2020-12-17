#####
# dataset: CIFAR-10
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# training: total 50000 images, divided into 5 batches, each with 10000 images
# (the entire training set contain exactly 5000 images from each class; some batch may contain more images from one
# class than other)
# test: total 10000 images in one batch (1000 randomly-selected images from each class)

# model: VGG16
# ref. paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Input, Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # NOTE: <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Load cifar10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the dataset
# Rescale/normalize the dataset to reduce variance, improve learning speed and performance
# Convert the pixel range from 0(black)-255(white) to 0-1
# *** vs. layers.experimental.preprocessing.Rescaling(scale=1./255)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Change the element type from float64 to float32 to reduce computational cost
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Convert scalar (0-9) to one-hot encoding
# tf.one_hot(indices, depth):
#     e.g. [[0],[1],[2]] --> [[[1,0,0],[0,1,0],[0,0,1]]]
# tf.squeeze(input, axis=1): Removes dimensions of size 1 from the shape of a tensor.
#     e.g. [[[1,0,0],[0,1,0],[0,0,1]]] (shape=1,3,3) --> [[1,0,0],[0,1,0],[0,0,1]] (shape=3,3)
y_train_onehot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_onehot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# Set the value of hyper-parameters
epochs = 20
learning_rate = 0.0001
batch_size = 16
# upsampling_size = (2,2)

# Results by hyper-parameters
# ==> learning rate: 0.001 (zigzag result of training acc)
# ==> learning rate: 0.0001; Epoch: 20; loss: 1.0698 - accuracy: 0.7885
# ==> learning rate: 0.00005; Epoch: 20; loss: 1.2411 - accuracy: 0.7474
# ==> learning rate: 0.0001; Epoch: 30; loss: 1.2150 - accuracy: 0.7998
# ==> learning rate: 0.0001; Epoch: 30; up-sampling: 2,2; loss: 0.9695 - accuracy: 0.8239
# ==> learning rate: 0.0001; Epoch: 20; batch size: 16;

# Initiate a VGG16 architecture
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')
# Rescale image (up-sampling) for better performance
# NOTE: change the next layer's input
# upsampling = tf.keras.layers.UpSampling2D(size=upsampling_size, name='upsampling')(input_tensor)

# block 1

# Conv2D:
# tf.keras.layers.Conv2D(filters, kernel_size, strides=(1,1), padding='valid',
#                       activation=None, kernel_initializer='glorot_uniform')
# This captures local connectivity by sliding a feature identifier, called filter or kernel, over the image.
# As more convolutional layers are added, higher level of features based on the previous detected features
# will be detected.
# A kernel refers to a 2D array of weights, while a filter is 3D structures of multiple kernels stacked together.
# Thus, the filter value would change the dimension of the output space (the last channel value).
# Likely, shifting the kernel by strides in each dimension over the image would result the reduction of image size.
# To keep the image dimension, set zero padding by padding='same'.
# Otherwise, if padding='valid', the output shape = (input_shape - pool_size + 1) / strides).
# *** check receptive field (convolution filter)

# Rectified Linear Unit (Relu) activation:
# It is a piecewise linear function that will output the input directly if it is positive, otherwise, zero.

# MaxPooling2D:
# tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')
# It downsamples the input representation by taking the maximum value over the window defined by pool_size
# for each dimension along the features axis.

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

# Fully connected (FC)

# Flatten:
# tf.keras.layers.Flatten
# Reshapes the input into a 1d array of elements while preserving the batch size.

# Dense:
# tf.keras.layers.Dense(units, activation=None)
# Fully connects the input nodes to the output nodes.
# Units = positive integer, dimensionality of the output space

# Softmax activation:
# It converts the logit scores to probabilities.

flatten = Flatten(name='flatten')(maxpool5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten) # unnecessary due to the final dimension size after block 5
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
fc3 = Dense(256, activation='relu', name='fc3')(flatten)  # NOTE: check input
output_tensor = Dense(10, activation='softmax', name='output')(fc3)

# Create a model
vgg16 = Model(input_tensor, output_tensor, name='vgg16')
vgg16.summary()  # plot the model architecture with the number of parameters (complexity)

# Compile the model

# Forward propagation:
# with randomly initialized weight and bias, propagates to the hidden units at each layer and finally produce output.
# Backward propagation:
# goes back from the cost backward through the network in order to update weights
# use loss function and gradient optimizer to compute gradient and update weight

# Loss function:
# tells how good a model is at making prediction at a given set of parameters
#   - loss: the difference between model prediction and actual answer
#   - e.g. Mean Square Error (MSE)
# cross entropy: uncertainty where the info can be incorrect; compare the probability and the actual value
# (if pred == actual, cross entropy = entropy; else cross entropy > entropy)
# sparse categorical cross entropy: with large number of classes, convert labels into one hot embedding of labels
# to speed up the execution

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

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # lower learning rate, better performance
vgg16.compile(loss='categorical_crossentropy',  # 'sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
# Epoch: running the entire dataset
vgg16.fit(x_train, y_train_onehot, batch_size=batch_size, epochs=epochs)

# Predict the model with test set
# model.evaluate(x, y): Returns the loss value & metrics values for the model in test mode.
# model.predict(x): Generates output predictions for the input samples
# The both methods use the same interpretation rules, but as the explanation indicates, model.evaluate() uses for
# validation, while model.predict() uses for prediction. Here, model.evaluate() is used to check the loss and
# accuracy easily.
vgg16.evaluate(x_test, y_test_onehot, verbose=2)

'''
# Build comparable vgg16 model with existing module
vgg16_module = tf.keras.applications.VGG16(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling='max',
    classes=10,
    classifier_activation='softmax'
)
vgg16_module.summary()

vgg16_module.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
# Epoch: running the entire dataset
vgg16_module.fit(x_train, y_train, epochs=1)

# Test the model with test set
vgg16_module.evaluate(x_test, y_test, verbose=2)
# ==> epoch: 1; loss: 2.3026 - accuracy: 0.1000
'''