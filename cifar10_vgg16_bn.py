#####
# dataset: CIFAR-10
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# training: total 50000 images, divided into 5 batches, each with 10000 images
# (the entire training set contain exactly 5000 images from each class; some batch may contain more images from one
# class than other)
# test: total 10000 images in one batch (1000 randomly-selected images from each class)

# model: VGG16
# ref. paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

# newly applied method: Batch Normalization (bn)

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Input, Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Load cifar10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Set the value of hyper-parameters
epochs = 30
learning_rate = 0.0001
batch_size = 64

# Results by hyper-parameters
# ==> learning rate: 0.0001; Epoch: 30; (default = None); loss: 1.2150 - accuracy: 0.7998
# ==> learning rate: 0.0001, epochs: 30; batch_size: 64; loss: 0.9947 - accuracy: 0.7937
# ==> learning rate: 0.0001, epochs: 30; batch_size: 128; loss: 1.0447 - accuracy: 0.7891 (faster)
# ==> learning rate: 0.0001, epochs: 30; batch_size: 256; running out of memory

# Batch the training set for batch normalization
# *** check shuffle(buffer_size).batch(batch_size)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Initiate a VGG16 architecture
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')

# Batch Normalization (BN)
# tf.keras.layers.BatchNormalization(axis=-1, trainable=True)
# Normalizes the activations of the previous layer at each batch, i.e. applies a transformation that maintains
# the mean activation close to 0 and the activation standard deviation close to 1.
# axis = Integer, the axis that should be normalized (1 if batch axis is first, -1 if batch axis is last)
# trainable = if True the variables will be marked as trainable and the layer will normalize its inputs using the mean
# and variance of the current batch of inputs

# *** Controversial issue regarding the order of convolution, BN, activation layers:
# Statistically, the "conv - relu - bn" order makes more sense, which eventually outputs zero mean and unit variance
# as its original purpose intended. However, the original paper prescribed using "conv - bn - relu" order and it is
# currently more widely used. Also, the impact of the order on the performance is negligible. Thus, in our projects,
# we will keep with the order of "conv - bn - relu".

# block 1
conv1_1 = Conv2D(64, 3, padding='same', name='conv1-1')(input_tensor)
conv1_1bn = BatchNormalization(axis=1, name='conv1_1bn')(conv1_1)
conv1_1relu = Activation('relu', name='conv1_1relu')(conv1_1bn)

conv1_2 = Conv2D(64, 3, padding='same', name='conv1-2')(conv1_1relu)
conv1_2bn = BatchNormalization(axis=1, name='conv1_2bn')(conv1_2)
conv1_2relu = Activation('relu', name='conv1_2relu')(conv1_2bn)

maxpool1 = MaxPooling2D(2, padding='same', name='maxpool1')(conv1_2relu)  # down-sampling # 16,16,64

# block 2
conv2_1 = Conv2D(128, 3, padding='same', name='conv2-1')(maxpool1)
conv2_1bn = BatchNormalization(axis=1, name='conv2_1bn')(conv2_1)
conv2_1relu = Activation('relu', name='conv2_1relu')(conv2_1bn)

conv2_2 = Conv2D(128, 3, padding='same', name='conv2-2')(conv2_1relu)
conv2_2bn = BatchNormalization(axis=1, name='conv2_2bn')(conv2_2)
conv2_2relu = Activation('relu', name='conv2_2relu')(conv2_2bn)

maxpool2 = MaxPooling2D(2, padding='same', name='maxpool2')(conv2_2relu)  # 8,8,128

# block 3
conv3_1 = Conv2D(256, 3, padding='same', name='conv3-1')(maxpool2)
conv3_1bn = BatchNormalization(axis=1, name='conv3_1bn')(conv3_1)
conv3_1relu = Activation('relu', name='conv3_1relu')(conv3_1bn)

conv3_2 = Conv2D(256, 3, padding='same', name='conv3-2')(conv3_1relu)
conv3_2bn = BatchNormalization(axis=1, name='conv3_2bn')(conv3_2)
conv3_2relu = Activation('relu', name='conv3_2relu')(conv3_2bn)

conv3_3 = Conv2D(256, 3, padding='same', name='conv3-3')(conv3_2relu)
conv3_3bn = BatchNormalization(axis=1, name='conv3_3bn')(conv3_3)
conv3_3relu = Activation('relu', name='conv3_3relu')(conv3_3bn)

maxpool3 = MaxPooling2D(2, padding='same', name='maxpool3')(conv3_3relu)  # 4,4,256

# block 4
conv4_1 = Conv2D(512, 3, padding='same', name='conv4-1')(maxpool3)
conv4_1bn = BatchNormalization(axis=1, name='conv4_1bn')(conv4_1)
conv4_1relu = Activation('relu', name='conv4_1relu')(conv4_1bn)

conv4_2 = Conv2D(512, 3, padding='same', name='conv4-2')(conv4_1relu)
conv4_2bn = BatchNormalization(axis=1, name='conv4_2bn')(conv4_2)
conv4_2relu = Activation('relu', name='conv4_2relu')(conv4_2bn)

conv4_3 = Conv2D(512, 3, padding='same', name='conv4-3')(conv4_2relu)
conv4_3bn = BatchNormalization(axis=1, name='conv4_3bn')(conv4_3)
conv4_3relu = Activation('relu', name='conv4_3relu')(conv4_3bn)

maxpool4 = MaxPooling2D(2, padding='same', name='maxpool4')(conv4_3relu)  # 2,2,512

# block 5
conv5_1 = Conv2D(512, 3, padding='same', name='conv5-1')(maxpool4)
conv5_1bn = BatchNormalization(axis=1, name='conv5_1bn')(conv5_1)
conv5_1relu = Activation('relu', name='conv5_1relu')(conv5_1bn)

conv5_2 = Conv2D(512, 3, padding='same', name='conv5-2')(conv5_1relu)
conv5_2bn = BatchNormalization(axis=1, name='conv5_2bn')(conv5_2)
conv5_2relu = Activation('relu', name='conv5_2relu')(conv5_2bn)

conv5_3 = Conv2D(512, 3, padding='same', name='conv5-3')(conv5_2relu)
conv5_3bn = BatchNormalization(axis=1, name='conv5_3bn')(conv5_3)
conv5_3relu = Activation('relu', name='conv5_3relu')(conv5_3bn)

maxpool5 = MaxPooling2D(2, padding='same', name='maxpool5')(conv5_3relu)  # 1,1,512

# FC
flatten = Flatten(name='flatten')(maxpool5)
# fc1 = Dense(4096, activation='relu', name='fc1')(flatten) # unnecessary due to the final dimension size after block 5
# fc2 = Dense(2048, activation='relu', name='fc2')(fc1)
fc3 = Dense(256, activation='relu', name='fc3')(flatten)  # NOTE: check input
output_tensor = Dense(10, activation='softmax', name='output')(fc3)

# Create a model
vgg16_bn = Model(input_tensor, output_tensor, name='vgg16_bn')
vgg16_bn.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # lower learning rate, better performance
vgg16_bn.compile(loss='sparse_categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
vgg16_bn.fit(x_train, y_train, batch_size=batch_size, epochs=30)  # Set batch

# Test the model with test set
vgg16_bn.predict(x_test, y_test, verbose=2)
