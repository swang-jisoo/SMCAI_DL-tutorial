#####
# Image Classification

# dataset: CIFAR-10
# description: a labeled set of colored images
# images: shape=(32, 32, 3)
# labels: num_classes=10 (0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, and 9=truck)
# total_num_examples=60000
# split: training=50000 (5000 images per class), test=10000 (1000 randomly-selected images per class)
# ref: https://www.cs.toronto.edu/~kriz/cifar.html

# model: VGG16 with Batch Normalization
# ref. paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
#####


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Input, Model

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
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=32; loss: 1.2150 - accuracy: 0.7998
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=64; loss: 0.9947 - accuracy: 0.7937
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=128; loss: 1.0447 - accuracy: 0.7891 (faster)
# ==> EPOCHS=30; LEARNING_RATE=0.0001; BATCH_SIZE=256; running out of memory


# Load the dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the dataset
# Data-type conversion
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0
# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, CLASSES)


# Batch the training set for batch normalization
# *** check shuffle(buffer_size).batch(batch_size)
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Build a VGG16 architecture
input_tensor = Input(shape=IMG_SHAPE, dtype='float32', name='input')

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
output_tensor = Dense(CLASSES, activation='softmax', name='output')(fc3)

# Create a model
vgg16_bn = Model(input_tensor, output_tensor, name='vgg16_bn')
vgg16_bn.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)  # lower learning rate, better performance
vgg16_bn.compile(loss='categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

# Fix the gpu memory issue (failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # NOTE: <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Train the model to adjust parameters to minimize the loss
vgg16_bn.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Test the model with test set
vgg16_bn.evaluate(x_test, y_test, verbose=VERBOSE)
