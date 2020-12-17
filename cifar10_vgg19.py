#####
# dataset: CIFAR-10
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# training: total 50000 images, divided into 5 batches, each with 10000 images
# (the entire training set contain exactly 5000 images from each class; some batch may contain more images from one
# class than other)
# test: total 10000 images in one batch (1000 randomly-selected images from each class)

# model: VGG19
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
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Set the value of hyper-parameters
epochs = 20
learning_rate = 0.0001
batch_size = 16
# upsampling_size = (2,2)

# Results by hyper-parameters
# ==> learning rate: 0.0001; Epoch: 20; batch size: 16;

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
conv3_4 = Conv2D(256, 3, activation='relu', padding='same', name='conv3-4')(conv3_3)
maxpool3 = MaxPooling2D(2, padding='same', name='maxpool3')(conv3_4)  # 4,4,256
# block 4
conv4_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-1')(maxpool3)
conv4_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-2')(conv4_1)
conv4_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-3')(conv4_2)
conv4_4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4-4')(conv4_3)
maxpool4 = MaxPooling2D(2, padding='same', name='maxpool4')(conv4_4)  # 2,2,512
# block 5
conv5_1 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-1')(maxpool4)
conv5_2 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-2')(conv5_1)
conv5_3 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-3')(conv5_2)
conv5_4 = Conv2D(512, 3, activation='relu', padding='same', name='conv5-4')(conv5_3)
maxpool5 = MaxPooling2D(2, padding='same', name='maxpool5')(conv5_4)  # 1,1,512

# Fully connected (FC)
flatten = Flatten(name='flatten')(maxpool5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
fc3 = Dense(256, activation='relu', name='fc3')(flatten)  # NOTE: check input
output_tensor = Dense(10, activation='softmax', name='output')(fc3)

# Create a model
vgg19 = Model(input_tensor, output_tensor, name='vgg19')
vgg19.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
vgg19.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
vgg19.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Predict the model with test set
vgg19.evaluate(x_test, y_test, verbose=2)
