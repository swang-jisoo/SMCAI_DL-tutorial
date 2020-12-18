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

# Transfer learning from a pretrained network
# ref.
# https://www.tensorflow.org/guide/keras/transfer_learning
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://blog.exxactcorp.com/discover-difference-deep-learning-training-inference/
# Idea behind transfer learning:
#   If a model is trained on a large and general enough dataset, it will effectively serve as a generic model
#   of other new, similar problems. This previously trained model is called a pre-trained model.
# General workflow:
#   1. Load in the pretrained base model (and pretrained weights)
#       - Take layers from a previously trained model
#       - Freeze them, so as to avoid destroying any of the information they contain during future training rounds
#         which will be used to extract meaningful features from new datasets
#         --> by setting layer.trainable = False:
#             Moves all the layer's weights from trainable (meant to be updated via gradient descent to minimize the
#             loss during training) to non-trainable (aren't meant to be updated during training)
#         *** vs. layer.__call__(training):
#             Controls whether the layer should run its forward pass in inference mode or training mode
#   2-1. Feature extraction; Run your new dataset through it and record the output of one (or several) layers from
#      the base model
#   Or 2-2. Stack the classification layers on top
#       - They will learn to turn the old features into predictions on a new dataset
#   3-1. Use that output as input data for a new, smaller model
#   Or 3-2. Train the new layers on your dataset
#   4. (Optional) Fine-tuning; Unfreeze the frozen base model
#       - Unfreeze a few of the top layers or the entire layers of a frozen model
#       - Jointly re-train both the newly-added classifier layers and the last layers of the base model on the new data
#         with a very low learning rate

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
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
y_train_onehot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_onehot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# Set the value of hyper-parameters
epochs = 20
learning_rate = 0.0001
batch_size = 16
# upsampling_size = (2,2)

# Results by hyper-parameters
# ==> learning rate: 0.0001; Epoch: 20; batch size: 16;

# Instantiate a base model with pre-trained weights
base_model = tf.keras.applications.VGG16(
    include_top=False,  # do not include the classifier at the top
    weights='imagenet',  # load weights pre-trained on ImageNet
    input_shape=(32, 32, 3),
)

# Initiate a VGG16 architecture
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')
# Rescale image (up-sampling) for better performance
# NOTE: change the next layer's input
# upsampling = tf.keras.layers.UpSampling2D(size=upsampling_size, name='upsampling')(input_tensor)

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

# Fully connected (FC)
flatten = Flatten(name='flatten')(maxpool5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten) # unnecessary due to the final dimension size after block 5
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
fc3 = Dense(256, activation='relu', name='fc3')(flatten)  # NOTE: check input
output_tensor = Dense(10, activation='softmax', name='output')(fc3)

# Create a model
vgg16 = Model(input_tensor, output_tensor, name='vgg16')
vgg16.summary()  # plot the model architecture with the number of parameters (complexity)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # lower learning rate, better performance
vgg16.compile(loss='categorical_crossentropy',  # 'sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
vgg16.fit(x_train, y_train_onehot, batch_size=batch_size, epochs=epochs)

# Predict the model with test set
vgg16.evaluate(x_test, y_test_onehot, verbose=2)
