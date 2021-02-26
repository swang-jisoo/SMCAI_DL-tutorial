#####
# Image Classification

# dataset: CIFAR-10
# description: a labeled set of colored images
# images: shape=(32, 32, 3)
# labels: num_classes=10 (0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, and 9=truck)
# total_num_examples=60000
# split: training=50000 (5000 images per class), test=10000 (1000 randomly-selected images per class)
# ref: https://www.cs.toronto.edu/~kriz/cifar.html

# model: VGG19
# ref. paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
#####


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Set the value of hyper-parameters
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
CLASSES = 10
EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
VERBOSE = 1

# Results by hyper-parameters
# ==> EPOCHS=20; LEARNING_RATE=0.0001; BATCH_SIZE=16; loss: 0.9532 - accuracy: 0.8017


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


# Build a VGG19 architecture
input_tensor = Input(shape=IMG_SHAPE, dtype='float32', name='input')

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
output_tensor = Dense(CLASSES, activation='softmax', name='output')(fc3)

# Create a model
vgg19 = Model(input_tensor, output_tensor, name='vgg19')
vgg19.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
vgg19.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Fix the gpu memory issue (failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # NOTE: <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Train the model to adjust parameters to minimize the loss
vgg19.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Predict the model with test set
vgg19.evaluate(x_test, y_test, verbose=VERBOSE)
prediction = vgg19.predict(x_test)
print(prediction)
