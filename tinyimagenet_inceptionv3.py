#####
# dataset: tinyImageNet
# All images are of size 64×64; 200 image classes
# Training = 100,000 images; Validation = 10,000 images; Test = 10,000 images.

# model: inception v3
# ref.
#   - Going deeper with convolutions (inception v1)
#   - Rethinking the Inception Architecture for Computer Vision (inception v2, v3)

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# General problem of Deep Neural Network
# ==> over-fitting & computationally expensive due to a large number of parameters
# Suggested solutions
# 1. Sparsely connected network / layer-by-layer (layer-wise) construction
#   - Opposed to fully connected network
#   - Cluster neurons with highly correlated outputs of the previous activation layer (Hebbian principle)
#   ==> Compared to dense matrix calculation (parallel computing), sparse matrix computation is less efficient
# 2. Inception architecture
#   - Intermediate step: sparsity at filter level + dense matrix computation
#   - The idea of sparsity is corresponded to a variation in the location of information in the image
#       ==> use a different size of kernels (larger one for global info; smaller for local)
#       ==> naive: [1x1 (cross-channel correlation)] + [3x3 + 5x5 (spatial correlation)] + 3x3maxpool
#   - To reduce computational cost:
#       ==> inception-v1: 1x1 + 1x1-3x3 + 1x1-5x5 + 3x3maxpool-1x1 -> filter concatenation
#       * 1x1 conv benefits: increased representational power of NN + dimension reduction (network-in-network)
#       ==> inception-v2,v3: 1x1 + 1x1-3x3 + 1x1-3x3-3x3 + 3x3maxpool-1x1 -> filter concatenation (factorization)

# RMSProp Optimizer; Factorized 7x7 convolutions; BatchNorm in the Auxillary Classifiers; Label Smoothing

# Import necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, Activation, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Set the value of hyper-parameters
upsampling_size = (2, 2)
learning_rate = 0.0001
epochs = 20
batch_size = 16

# Load tiny imangenet dataset

# To download the imagenet, you may need to install wget and unzip first.
# For installation, check here:
# https://builtvisible.com/download-your-website-with-wget/
# https://zoomadmin.com/HowToInstall/UbuntuPackage/unzip
# On terminal, run the codes below:
# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# unzip -qq 'tiny-imagenet-200.zip'

path = 'C:/wgetdown/tiny-imagenet-200'

val_data = pd.read_csv('./tiny-imagenet-200/val/val_annotations.txt', sep='\t', header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])
val_data.drop(['X', 'Y', 'H', 'W'], axis=1, inplace=True)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(r'./tiny-imagenet-200/train/', target_size=(64, 64),
                                                    color_mode='rgb', class_mode='categorical',
                                                    batch_size=batch_size, shuffle=True, seed=42)
valid_generator = valid_datagen.flow_from_dataframe(val_data, directory='./tiny-imagenet-200/val/images/',
                                                    x_col='File', y_col='Class', target_size=(64, 64),
                                                    color_mode='rgb', class_mode='categorical',
                                                    batch_size=batch_size, shuffle=True, seed=42)

# Results by hyper-parameters
# ==> upsampling: 2; learing rate: 0.0001; epoch: 20; batch size: 16;

# Initiate a ResNet50 architecture
input_tensor = Input(shape=(64, 64, 3), dtype='float32', name='input')
# Rescale image (up-sampling) for better performance
upsampling = tf.keras.layers.UpSampling2D(size=upsampling_size, name='upsampling')(input_tensor)

# conv1
conv1 = Conv2D(32, 3, strides=(2, 2), name='conv1')(upsampling)
conv1 = Conv2D(32, 3, padding='SAME', name='conv1')(conv1)
conv1_bn = BatchNormalization(axis=1, name='conv1_bn')(x)
x = Conv2D(64, 3, padding='SAME', name='conv1')(conv1_bn)

'''
# tf.keras module
incepv3_module = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(75, 75, 3),
    pooling='max',
    classes=10,
    classifier_activation="softmax",
)

incepv3_module.summary()

incepv3_module.compile(optimizer='rmsprop',
                       loss={'predictions': 'categorical_crossentropy',
                             'aux_classifier': 'categorical_crossentropy'},
                       loss_weights={'predictions': 1., 'aux_classifier': 0.2},
                       metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
# NOTE: Need to resize the dataset
incepv3_module.fit(x_train, {'predictions': y_train, 'aux_classifier': y_train},
                   batch_size=16, epochs=1)

# Test the model with test set
incepv3_module.evaluate(x_test, y_test, verbose=2)
# ==> learning rate: 0.001 (default); epoch: 1;
# ==> learning rate: 0.001 (default); epoch: 20;
'''
