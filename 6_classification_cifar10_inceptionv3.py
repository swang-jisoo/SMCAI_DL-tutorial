#####
# dataset:
# 1) CIFAR-10
# 2) tinyImageNet
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
import numpy as np
#from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, concatenate

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # <=0.3 with GTX 1050 (2GB)
session = tf.compat.v1.Session(config=config)

# Set the value of hyper-parameters
upsampling_size = (3, 3)
learning_rate = 0.0001
epochs = 20
batch_size = 16

# Load cifar10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Load tiny imagenet dataset

# To download the imagenet, you may need to install wget and unzip first.
# For installation, check here:
# https://builtvisible.com/download-your-website-with-wget/
# https://zoomadmin.com/HowToInstall/UbuntuPackage/unzip
# On terminal, run the codes below:
# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# unzip -qq 'tiny-imagenet-200.zip'

'''
path = 'C:/wgetdown/tiny-imagenet-200'

val_data = pd.read_csv(path+'/tiny-imagenet-200/val/val_annotations.txt', sep='\t', header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])
val_data.drop(['X', 'Y', 'H', 'W'], axis=1, inplace=True)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(path+'/tiny-imagenet-200/train/', target_size=(64, 64),
                                                    color_mode='rgb', class_mode='categorical',
                                                    batch_size=batch_size, shuffle=True, seed=42)
valid_generator = valid_datagen.flow_from_dataframe(val_data, directory=path+'/tiny-imagenet-200/val/images/',
                                                    x_col='File', y_col='Class', target_size=(64, 64),
                                                    color_mode='rgb', class_mode='categorical',
                                                    batch_size=batch_size, shuffle=True, seed=42)
'''

# Results by hyper-parameters
# ==> upsampling: 2; learing rate: 0.0001; epoch: 20; batch size: 16; loss: 0.8499 - accuracy: 0.7981

# Initiate a ResNet50 architecture
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')  # 75,75,3
# Rescale image (up-sampling) for better performance
upsampling = tf.keras.layers.UpSampling2D(size=upsampling_size, name='upsampling')(input_tensor)

# conv1
conv1_conv = Conv2D(32, 3, strides=(2, 2),
                    kernel_initializer='he_normal', name='conv1_conv')(upsampling)  # 37,37,32
conv1_bn = BatchNormalization(axis=1, name='conv1_bn')(conv1_conv)
conv1_relu = Activation('relu', name='conv1_relu')(conv1_bn)

# conv2_1
conv2_1conv = Conv2D(32, 3,
                     kernel_initializer='he_normal', name='conv2_1conv')(conv1_relu)  # 35,35,32
conv2_1bn = BatchNormalization(axis=1, name='conv2_1bn')(conv2_1conv)
conv2_1relu = Activation('relu', name='conv2_1relu')(conv2_1bn)

# conv2_2
conv2_2conv = Conv2D(64, 3, padding='SAME',
                     kernel_initializer='he_normal', name='conv2_2conv')(conv2_1relu)  # 35,35,64
conv2_2bn = BatchNormalization(axis=1, name='conv2_2bn')(conv2_2conv)
conv2_2relu = Activation('relu', name='conv2_2relu')(conv2_2bn)

# maxpool1
maxpool1 = MaxPooling2D((3, 3), strides=2, name='maxpool1')(conv2_2relu)  # 17,17,64

# conv2_3
conv2_3conv = Conv2D(80, 1,
                     kernel_initializer='he_normal', name='conv2_3conv')(maxpool1)  # 17,17,80
conv2_3bn = BatchNormalization(axis=1, name='conv2_3bn')(conv2_3conv)
conv2_3relu = Activation('relu', name='conv2_3relu')(conv2_3bn)

# conv2_4
conv2_4conv = Conv2D(192, 3,
                     kernel_initializer='he_normal', name='conv2_4conv')(conv2_3relu)  # 15,15,192
conv2_4bn = BatchNormalization(axis=1, name='conv2_4bn')(conv2_4conv)
conv2_4relu = Activation('relu', name='conv2_4relu')(conv2_4bn)

# maxpool1
maxpool2 = MaxPooling2D((3, 3), strides=2, name='maxpool2')(conv2_4relu)  # 7,7,192

# Inception1
# 1x1
inception1_1x1conv = Conv2D(64, 1, name='inception1_1x1conv')(maxpool2)  # 7,7,64
inception1_1x1bn = BatchNormalization(axis=1, name='inception1_1x1bn')(inception1_1x1conv)
inception1_1x1relu = Activation('relu', name='inception1_1x1relu')(inception1_1x1bn)
# 1x1-3x3
# *** 5x5 --> 3x3?
inception1_3x3conv1 = Conv2D(48, 1, name='inception1_3x3conv1')(maxpool2)  # 7,7,48
inception1_3x3bn1 = BatchNormalization(axis=1, name='inception1_3x3bn1')(inception1_3x3conv1)
inception1_3x3relu1 = Activation('relu', name='inception1_3x3relu1')(inception1_3x3bn1)
inception1_3x3conv2 = Conv2D(64, 3, padding='SAME', name='inception1_3x3conv2')(inception1_3x3relu1)  # 7,7,64
inception1_3x3bn2 = BatchNormalization(axis=1, name='inception1_3x3bn2')(inception1_3x3conv2)
inception1_3x3relu2 = Activation('relu', name='inception1_3x3relu2')(inception1_3x3bn2)
# 1x1-3x3-3x3
inception1_5x5conv1 = Conv2D(48, 1, name='inception1_5x5conv1')(maxpool2)  # 7,7,48
inception1_5x5bn1 = BatchNormalization(axis=1, name='inception1_5x5bn1')(inception1_5x5conv1)
inception1_5x5relu1 = Activation('relu', name='inception1_5x5relu1')(inception1_5x5bn1)
inception1_5x5conv2 = Conv2D(64, 3, padding='SAME', name='inception1_5x5conv2')(inception1_5x5relu1)  # 7,7,64
inception1_5x5bn2 = BatchNormalization(axis=1, name='inception1_5x5bn2')(inception1_5x5conv2)
inception1_5x5relu2 = Activation('relu', name='inception1_5x5relu2')(inception1_5x5bn2)
inception1_5x5conv3 = Conv2D(64, 3, padding='SAME', name='inception1_5x5conv3')(inception1_5x5relu2)  # 7,7,64
inception1_5x5bn3 = BatchNormalization(axis=1, name='inception1_5x5bn3')(inception1_5x5conv3)
inception1_5x5relu3 = Activation('relu', name='inception1_5x5relu3')(inception1_5x5bn3)
# avgpool
inception1_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                      name='inception1_avgpool')(maxpool2)  # 7,7,192
inception1_avgpool_conv = Conv2D(64, 1, name='inception1_avgpool_conv')(inception1_avgpool)  # 7,7,48
inception1_avgpool_bn = BatchNormalization(axis=1, name='inception1_avgpool_bn')(inception1_avgpool_conv)
inception1_avgpool_relu = Activation('relu', name='inception1_avgpool_relu')(inception1_avgpool_bn)

inception1_concat = concatenate([inception1_1x1relu, inception1_3x3relu2, inception1_5x5relu3, inception1_avgpool_relu],
                                axis=-1, name='inception1_concat')  # 7,7,256 (64*4)

# inception2
# 1x1
inception2_1x1conv = Conv2D(64, 1, name='inception2_1x1conv')(inception1_concat)  # 7,7,64
inception2_1x1bn = BatchNormalization(axis=1, name='inception2_1x1bn')(inception2_1x1conv)
inception2_1x1relu = Activation('relu', name='inception2_1x1relu')(inception2_1x1bn)
# 1x1-3x3
inception2_3x3conv1 = Conv2D(48, 1, name='inception2_3x3conv1')(inception1_concat)  # 7,7,48
inception2_3x3bn1 = BatchNormalization(axis=1, name='inception2_3x3bn1')(inception2_3x3conv1)
inception2_3x3relu1 = Activation('relu', name='inception2_3x3relu1')(inception2_3x3bn1)
inception2_3x3conv2 = Conv2D(64, 3, padding='SAME', name='inception2_3x3conv2')(inception2_3x3relu1)  # 7,7,64
inception2_3x3bn2 = BatchNormalization(axis=1, name='inception2_3x3bn2')(inception2_3x3conv2)
inception2_3x3relu2 = Activation('relu', name='inception2_3x3relu2')(inception2_3x3bn2)
# 1x1-3x3-3x3
inception2_5x5conv1 = Conv2D(48, 1, name='inception2_5x5conv1')(inception1_concat)  # 7,7,48
inception2_5x5bn1 = BatchNormalization(axis=1, name='inception2_5x5bn1')(inception2_5x5conv1)
inception2_5x5relu1 = Activation('relu', name='inception2_5x5relu1')(inception2_5x5bn1)
inception2_5x5conv2 = Conv2D(64, 3, padding='SAME', name='inception2_5x5conv2')(inception2_5x5relu1)  # 7,7,64
inception2_5x5bn2 = BatchNormalization(axis=1, name='inception2_5x5bn2')(inception2_5x5conv2)
inception2_5x5relu2 = Activation('relu', name='inception2_5x5relu2')(inception2_5x5bn2)
inception2_5x5conv3 = Conv2D(64, 3, padding='SAME', name='inception2_5x5conv3')(inception2_5x5relu2)  # 7,7,64
inception2_5x5bn3 = BatchNormalization(axis=1, name='inception2_5x5bn3')(inception2_5x5conv3)
inception2_5x5relu3 = Activation('relu', name='inception2_5x5relu3')(inception2_5x5bn3)
# avgpool
inception2_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                      name='inception2_avgpool')(inception1_concat)  # 7,7,256
inception2_avgpool_conv = Conv2D(64, 1, name='inception2_avgpool_conv')(inception2_avgpool)  # 7,7,64
inception2_avgpool_bn = BatchNormalization(axis=1, name='inception2_avgpool_bn')(inception2_avgpool_conv)
inception2_avgpool_relu = Activation('relu', name='inception2_avgpool_relu')(inception2_avgpool_bn)

inception2_concat = concatenate([inception2_1x1relu, inception2_3x3relu2, inception2_5x5relu3, inception2_avgpool_relu],
                                axis=-1, name='inception2_concat')  # 7,7,256 (64*4)

# inception3
# different structure compared to inception2
# 1x1
# *** dimension value
inception3_1x1conv = Conv2D(384, 3, strides=(2, 2), name='inception3_1x1conv')(inception2_concat)  # 3,3,384
inception3_1x1bn = BatchNormalization(axis=1, name='inception3_1x1bn')(inception3_1x1conv)
inception3_1x1relu = Activation('relu', name='inception3_1x1relu')(inception3_1x1bn)
# 1x1-3x3-3x3
inception3_5x5conv1 = Conv2D(64, 1, name='inception3_5x5conv1')(inception2_concat)  # 7,7,64
inception3_5x5bn1 = BatchNormalization(axis=1, name='inception3_5x5bn1')(inception3_5x5conv1)
inception3_5x5relu1 = Activation('relu', name='inception3_5x5relu1')(inception3_5x5bn1)
inception3_5x5conv2 = Conv2D(96, 3, padding='SAME', name='inception3_5x5conv2')(inception3_5x5relu1)  # 7,7,96
inception3_5x5bn2 = BatchNormalization(axis=1, name='inception3_5x5bn2')(inception3_5x5conv2)
inception3_5x5relu2 = Activation('relu', name='inception3_5x5relu2')(inception3_5x5bn2)
inception3_5x5conv3 = Conv2D(96, 3, strides=(2, 2), name='inception3_5x5conv3')(inception3_5x5relu2)  # 3,3,96
inception3_5x5bn3 = BatchNormalization(axis=1, name='inception3_5x5bn3')(inception3_5x5conv3)
inception3_5x5relu3 = Activation('relu', name='inception3_5x5relu3')(inception3_5x5bn3)
# avgpool
# *** avg pool --> max pool
inception3_maxpool = MaxPooling2D((3, 3), strides=(2, 2),
                                  name='inception3_maxpool')(inception2_concat)  # 3,3,256
# *** no conv

inception3_concat = concatenate([inception3_1x1relu, inception3_5x5relu3, inception3_maxpool],
                                axis=-1, name='inception3_concat')  # 3,3,736 (384+96+256)

# inception4
# 1x1
inception4_1x1conv = Conv2D(192, 1, name='inception4_1x1conv')(inception3_concat)  # 3,3,192
inception4_1x1bn = BatchNormalization(axis=1, name='inception4_1x1bn')(inception4_1x1conv)
inception4_1x1relu = Activation('relu', name='inception4_1x1relu')(inception4_1x1bn)
# 1x1-1x7-7x1
inception4_1771conv1 = Conv2D(128, 1, name='inception4_1771conv1')(inception3_concat)  # 3,3,128
inception4_1771bn1 = BatchNormalization(axis=1, name='inception4_1771bn1')(inception4_1771conv1)
inception4_1771relu1 = Activation('relu', name='inception4_1771relu1')(inception4_1771bn1)
inception4_1771conv2 = Conv2D(128, (1, 7), padding='SAME', name='inception4_1771conv2')(inception4_1771relu1)  # 3,3,128
inception4_1771bn2 = BatchNormalization(axis=1, name='inception4_1771bn2')(inception4_1771conv2)
inception4_1771relu2 = Activation('relu', name='inception4_1771relu2')(inception4_1771bn2)
inception4_1771conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception4_1771conv3')(inception4_1771relu2)  # 3,3,192
inception4_1771bn3 = BatchNormalization(axis=1, name='inception4_1771bn3')(inception4_1771conv3)
inception4_1771relu3 = Activation('relu', name='inception4_1771relu3')(inception4_1771bn3)
# 1x1-7x1-1x7-7x1-1x7
inception4_7117conv1 = Conv2D(128, 1, name='inception4_7117conv1')(inception3_concat)  # 3,3,128
inception4_7117bn1 = BatchNormalization(axis=1, name='inception4_7117bn1')(inception4_7117conv1)
inception4_7117relu1 = Activation('relu', name='inception4_7117relu1')(inception4_7117bn1)
inception4_7117conv2 = Conv2D(128, (7, 1), padding='SAME', name='inception4_7117conv2')(inception4_7117relu1)  # 3,3,128
inception4_7117bn2 = BatchNormalization(axis=1, name='inception4_7117bn2')(inception4_7117conv2)
inception4_7117relu2 = Activation('relu', name='inception4_7117relu2')(inception4_7117bn2)
inception4_7117conv3 = Conv2D(128, (1, 7), padding='SAME', name='inception4_7117conv3')(inception4_7117relu2)  # 3,3,128
inception4_7117bn3 = BatchNormalization(axis=1, name='inception4_7117bn3')(inception4_7117conv3)
inception4_7117relu3 = Activation('relu', name='inception4_7117relu3')(inception4_7117bn3)
inception4_7117conv4 = Conv2D(128, (7, 1), padding='SAME', name='inception4_7117conv4')(inception4_7117relu3)  # 3,3,128
inception4_7117bn4 = BatchNormalization(axis=1, name='inception4_7117bn4')(inception4_7117conv4)
inception4_7117relu4 = Activation('relu', name='inception4_7117relu4')(inception4_7117bn4)
inception4_7117conv5 = Conv2D(192, (1, 7), padding='SAME', name='inception4_7117conv5')(inception4_7117relu4)  # 3,3,192
inception4_7117bn5 = BatchNormalization(axis=1, name='inception4_7117bn5')(inception4_7117conv5)
inception4_7117relu5 = Activation('relu', name='inception4_7117relu5')(inception4_7117bn5)
# avgpool
inception4_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                      name='inception4_avgpool')(inception3_concat)  # 3,3,736
inception4_avgpool_conv = Conv2D(192, 1, name='inception4_avgpool_conv')(inception4_avgpool)  # 3,3,192
inception4_avgpool_bn = BatchNormalization(axis=1, name='inception4_avgpool_bn')(inception4_avgpool_conv)
inception4_avgpool_relu = Activation('relu', name='inception4_avgpool_relu')(inception4_avgpool_bn)

inception4_concat = concatenate([inception4_1x1relu, inception4_1771relu3,
                                 inception4_7117relu5, inception4_avgpool_relu],
                                axis=-1, name='inception4_concat')  # 3,3,768 (192*4)

x = inception4_concat
# inception5, 6
for r in range(5, 7):
    i = str(r)
    # 1x1
    inception5_1x1conv = Conv2D(192, 1,
                                name='inception'+i+'_1x1conv')(x)  # 3,3,192
    inception5_1x1bn = BatchNormalization(axis=1,
                                          name='inception'+i+'_1x1bn')(inception5_1x1conv)
    inception5_1x1relu = Activation('relu',
                                    name='inception'+i+'_1x1relu')(inception5_1x1bn)
    # 1x1-1x7-7x1
    inception5_1771conv1 = Conv2D(160, 1,
                                  name='inception'+i+'_1771conv1')(x)  # 3,3,160
    inception5_1771bn1 = BatchNormalization(axis=1,
                                            name='inception'+i+'_1771bn1')(inception5_1771conv1)
    inception5_1771relu1 = Activation('relu',
                                      name='inception'+i+'_1771relu1')(inception5_1771bn1)
    inception5_1771conv2 = Conv2D(160, (1, 7), padding='SAME',
                                  name='inception'+i+'_1771conv2')(inception5_1771relu1)  # 3,3,160
    inception5_1771bn2 = BatchNormalization(axis=1,
                                            name='inception'+i+'_1771bn2')(inception5_1771conv2)
    inception5_1771relu2 = Activation('relu', name='inception'+i+'_1771relu2')(inception5_1771bn2)
    inception5_1771conv3 = Conv2D(192, (7, 1), padding='SAME',
                                  name='inception'+i+'_1771conv3')(inception5_1771relu2)  # 3,3,192
    inception5_1771bn3 = BatchNormalization(axis=1,
                                            name='inception'+i+'_1771bn3')(inception5_1771conv3)
    inception5_1771relu3 = Activation('relu',
                                      name='inception'+i+'_1771relu3')(inception5_1771bn3)
    # 1x1-7x1-1x7-7x1-1x7
    inception5_7117conv1 = Conv2D(160, 1,
                                  name='inception'+i+'_7117conv1')(x)  # 3,3,160
    inception5_7117bn1 = BatchNormalization(axis=1,
                                            name='inception'+i+'_7117bn1')(inception5_7117conv1)
    inception5_7117relu1 = Activation('relu',
                                      name='inception'+i+'_7117relu1')(inception5_7117bn1)
    inception5_7117conv2 = Conv2D(160, (7, 1), padding='SAME',
                                  name='inception'+i+'_7117conv2')(inception5_7117relu1)  # 3,3,160
    inception5_7117bn2 = BatchNormalization(axis=1,
                                            name='inception'+i+'_7117bn2')(inception5_7117conv2)
    inception5_7117relu2 = Activation('relu',
                                      name='inception'+i+'_7117relu2')(inception5_7117bn2)
    inception5_7117conv3 = Conv2D(160, (1, 7), padding='SAME',
                                  name='inception'+i+'_7117conv3')(inception5_7117relu2)  # 3,3,160
    inception5_7117bn3 = BatchNormalization(axis=1,
                                            name='inception'+i+'_7117bn3')(inception5_7117conv3)
    inception5_7117relu3 = Activation('relu',
                                      name='inception'+i+'_7117relu3')(inception5_7117bn3)
    inception5_7117conv4 = Conv2D(160, (7, 1), padding='SAME',
                                  name='inception'+i+'_7117conv4')(inception5_7117relu3)  # 3,3,160
    inception5_7117bn4 = BatchNormalization(axis=1,
                                            name='inception'+i+'_7117bn4')(inception5_7117conv4)
    inception5_7117relu4 = Activation('relu',
                                      name='inception'+i+'_7117relu4')(inception5_7117bn4)
    inception5_7117conv5 = Conv2D(192, (1, 7), padding='SAME',
                                  name='inception'+i+'_7117conv5')(inception5_7117relu4)  # 3,3,192
    inception5_7117bn5 = BatchNormalization(axis=1,
                                            name='inception'+i+'_7117bn5')(inception5_7117conv5)
    inception5_7117relu5 = Activation('relu',
                                      name='inception'+i+'_7117relu5')(inception5_7117bn5)
    # avgpool
    inception5_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                          name='inception'+i+'_avgpool')(x)  # 3,3,768
    inception5_avgpool_conv = Conv2D(192, 1,
                                     name='inception'+i+'_avgpool_conv')(inception5_avgpool)  # 3,3,192
    inception5_avgpool_bn = BatchNormalization(axis=1,
                                               name='inception'+i+'_avgpool_bn')(inception5_avgpool_conv)
    inception5_avgpool_relu = Activation('relu',
                                         name='inception'+i+'_avgpool_relu')(inception5_avgpool_bn)

    x = concatenate([inception5_1x1relu, inception5_1771relu3, inception5_7117relu5, inception5_avgpool_relu],
                    axis=-1, name='inception'+i+'_concat')  # 3,3,768 (192*4)
inception6_concat = x

# inception7
# 1x1
inception7_1x1conv = Conv2D(192, 1, name='inception7_1x1conv')(inception6_concat)  # 3,3,192
inception7_1x1bn = BatchNormalization(axis=1, name='inception7_1x1bn')(inception7_1x1conv)
inception7_1x1relu = Activation('relu', name='inception7_1x1relu')(inception7_1x1bn)
# 1x1-1x7-7x1
inception7_1771conv1 = Conv2D(192, 1, name='inception7_1771conv1')(inception6_concat)  # 3,3,192
inception7_1771bn1 = BatchNormalization(axis=1, name='inception7_1771bn1')(inception7_1771conv1)
inception7_1771relu1 = Activation('relu', name='inception7_1771relu1')(inception7_1771bn1)
inception7_1771conv2 = Conv2D(192, (1, 7), padding='SAME', name='inception7_1771conv2')(inception7_1771relu1)  # 3,3,192
inception7_1771bn2 = BatchNormalization(axis=1, name='inception7_1771bn2')(inception7_1771conv2)
inception7_1771relu2 = Activation('relu', name='inception7_1771relu2')(inception7_1771bn2)
inception7_1771conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception7_1771conv3')(inception7_1771relu2)  # 3,3,192
inception7_1771bn3 = BatchNormalization(axis=1, name='inception7_1771bn3')(inception7_1771conv3)
inception7_1771relu3 = Activation('relu', name='inception7_1771relu3')(inception7_1771bn3)
# 1x1-7x1-1x7-7x1-1x7
inception7_7117conv1 = Conv2D(192, 1, name='inception7_7117conv1')(inception6_concat)  # 3,3,192
inception7_7117bn1 = BatchNormalization(axis=1, name='inception7_7117bn1')(inception7_7117conv1)
inception7_7117relu1 = Activation('relu', name='inception7_7117relu1')(inception7_7117bn1)
inception7_7117conv2 = Conv2D(192, (7, 1), padding='SAME', name='inception7_7117conv2')(inception7_7117relu1)  # 3,3,192
inception7_7117bn2 = BatchNormalization(axis=1, name='inception7_7117bn2')(inception7_7117conv2)
inception7_7117relu2 = Activation('relu', name='inception7_7117relu2')(inception7_7117bn2)
inception7_7117conv3 = Conv2D(192, (1, 7), padding='SAME', name='inception7_7117conv3')(inception7_7117relu2)  # 3,3,192
inception7_7117bn3 = BatchNormalization(axis=1, name='inception7_7117bn3')(inception7_7117conv3)
inception7_7117relu3 = Activation('relu', name='inception7_7117relu3')(inception7_7117bn3)
inception7_7117conv4 = Conv2D(192, (7, 1), padding='SAME', name='inception7_7117conv4')(inception7_7117relu3)  # 3,3,192
inception7_7117bn4 = BatchNormalization(axis=1, name='inception7_7117bn4')(inception7_7117conv4)
inception7_7117relu4 = Activation('relu', name='inception7_7117relu4')(inception7_7117bn4)
inception7_7117conv5 = Conv2D(192, (1, 7), padding='SAME', name='inception7_7117conv5')(inception7_7117relu4)  # 3,3,192
inception7_7117bn5 = BatchNormalization(axis=1, name='inception7_7117bn5')(inception7_7117conv5)
inception7_7117relu5 = Activation('relu', name='inception7_7117relu5')(inception7_7117bn5)
# avgpool
inception7_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                      name='inception7_avgpool')(inception6_concat)  # 3,3,768
inception7_avgpool_conv = Conv2D(192, 1, name='inception7_avgpool_conv')(inception7_avgpool)  # 3,3,192
inception7_avgpool_bn = BatchNormalization(axis=1, name='inception7_avgpool_bn')(inception7_avgpool_conv)
inception7_avgpool_relu = Activation('relu', name='inception7_avgpool_relu')(inception7_avgpool_bn)

inception7_cocat = concatenate([inception7_1x1relu, inception7_1771relu3,
                                inception7_7117relu5, inception7_avgpool_relu],
                               axis=-1, name='inception7_cocat')  # 3,3,768 (192*4)

# inception8
# 1x1-3x3
inception8_3x3conv1 = Conv2D(192, 1, name='inception8_3x3conv1')(inception7_cocat)  # 3,3,192
inception8_3x3bn1 = BatchNormalization(axis=1, name='inception8_3x3bn1')(inception8_3x3conv1)
inception8_3x3relu1 = Activation('relu', name='inception8_3x3relu1')(inception8_3x3bn1)
inception8_3x3conv2 = Conv2D(320, 3, strides=(2, 2), name='inception8_3x3conv2')(inception8_3x3relu1)  # 1,1,320
inception8_3x3bn2 = BatchNormalization(axis=1, name='inception8_3x3bn2')(inception8_3x3conv2)
inception8_3x3relu2 = Activation('relu', name='inception8_3x3relu2')(inception8_3x3bn2)
# 1x1-1x7-7x1
inception8_1771conv1 = Conv2D(192, 1, name='inception8_1771conv1')(inception7_cocat)  # 3,3,192
inception8_1771bn1 = BatchNormalization(axis=1, name='inception8_1771bn1')(inception8_1771conv1)
inception8_1771relu1 = Activation('relu', name='inception8_1771relu1')(inception8_1771bn1)
inception8_1771conv2 = Conv2D(192, (1, 7), padding='SAME', name='inception8_1771conv2')(inception8_1771relu1)  # 3,3,192
inception8_1771bn2 = BatchNormalization(axis=1, name='inception8_1771bn2')(inception8_1771conv2)
inception8_1771relu2 = Activation('relu', name='inception8_1771relu2')(inception8_1771bn2)
inception8_1771conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception8_1771conv3')(inception8_1771relu2)  # 3,3,192
inception8_1771bn3 = BatchNormalization(axis=1, name='inception8_1771bn3')(inception8_1771conv3)
inception8_1771relu3 = Activation('relu', name='inception8_1771relu3')(inception8_1771bn3)
inception8_1771conv4 = Conv2D(192, 3, strides=(2, 2), name='inception8_1771conv4')(inception8_1771relu3)  # 1,1,192
inception8_1771bn4 = BatchNormalization(axis=1, name='inception8_1771bn4')(inception8_1771conv4)
inception8_1771relu4 = Activation('relu', name='inception8_1771relu4')(inception8_1771bn4)
# avgpool
inception8_maxpool = MaxPooling2D((3, 3), strides=(2, 2),
                                  name='inception8_maxpool')(inception7_cocat)  # 1,1,768

inception8_concat = concatenate([inception8_3x3relu2, inception8_1771relu4, inception8_maxpool],
                                axis=-1, name='inception8_concat')  # 1,1,1280 (320+192+768)

x = inception8_concat
# inception9. 10
for r in range(9, 11):
    i = str(r)
    # 1x1
    inception9_1x1conv = Conv2D(320, 1, name='inception' + i + '_1x1conv')(x)  # 1,1,320
    inception9_1x1bn = BatchNormalization(axis=1, name='inception' + i + '_1x1bn')(inception9_1x1conv)
    inception9_1x1relu = Activation('relu', name='inception' + i + '_1x1relu')(inception9_1x1bn)

    # 1x1-1x3-3x1-concat
    inception9_1331conv1 = Conv2D(384, 1,
                                  name='inception' + i + '_1331conv1')(x)  # 1,1,384
    inception9_1331bn1 = BatchNormalization(axis=1,
                                            name='inception' + i + '_1331bn1')(inception9_1331conv1)
    inception9_1331relu1 = Activation('relu',
                                      name='inception' + i + '_1331relu1')(inception9_1331bn1)
    inception9_1331conv2 = Conv2D(384, (1, 3), padding='SAME',
                                  name='inception' + i + '_1331conv2')(inception9_1331relu1)  # 1,1,384
    inception9_1331bn2 = BatchNormalization(axis=1,
                                            name='inception' + i + '_1331bn2')(inception9_1331conv2)
    inception9_1331relu2 = Activation('relu',
                                      name='inception' + i + '_1331relu2')(inception9_1331bn2)
    inception9_1331conv3 = Conv2D(384, (3, 1), padding='SAME',
                                  name='inception' + i + '_1331conv3')(inception9_1331relu1)  # 1,1,384
    inception9_1331bn3 = BatchNormalization(axis=1,
                                            name='inception' + i + '_1331bn3')(inception9_1331conv3)
    inception9_1331relu3 = Activation('relu',
                                      name='inception' + i + '_1331relu3')(inception9_1331bn3)
    inception9_1331concat = concatenate([inception9_1331relu2, inception9_1331relu3],
                                        axis=-1, name='inception' + i + '_1331concat')  # 1,1,768 (384*2)

    # 1x1-1x3-3x1-concat
    inception9_331331conv1 = Conv2D(448, 1,
                                  name='inception' + i + '_331331conv1')(x)  # 1,1,448
    inception9_331331bn1 = BatchNormalization(axis=1,
                                            name='inception' + i + '_331331bn1')(inception9_331331conv1)
    inception9_331331relu1 = Activation('relu',
                                      name='inception' + i + '_331331relu1')(inception9_331331bn1)
    inception9_331331conv2 = Conv2D(384, 3, padding='SAME',
                                  name='inception' + i + '_331331conv2')(inception9_331331relu1)  # 1,1,384
    inception9_331331bn2 = BatchNormalization(axis=1,
                                            name='inception' + i + '_331331bn2')(inception9_331331conv2)
    inception9_331331relu2 = Activation('relu',
                                      name='inception' + i + '_331331relu2')(inception9_331331bn2)
    inception9_331331conv3 = Conv2D(384, (1, 3), padding='SAME',
                                  name='inception' + i + '_331331conv3')(inception9_331331relu2)  # 1,1,384
    inception9_331331bn3 = BatchNormalization(axis=1,
                                            name='inception' + i + '_331331bn3')(inception9_331331conv3)
    inception9_331331relu3 = Activation('relu',
                                      name='inception' + i + '_331331relu3')(inception9_331331bn3)
    inception9_331331conv4 = Conv2D(384, (3, 1), padding='SAME',
                                  name='inception' + i + '_331331conv4')(inception9_331331relu2)  # 1,1,384
    inception9_331331bn4 = BatchNormalization(axis=1,
                                            name='inception' + i + '_331331bn4')(inception9_331331conv4)
    inception9_331331relu4 = Activation('relu',
                                      name='inception' + i + '_331331relu4')(inception9_331331bn4)
    inception9_331331concat = concatenate([inception9_331331relu3, inception9_331331relu4],
                                        axis=-1, name='inception' + i + '_331331concat')  # 1,1,768 (384*2)

    # avgpool
    inception9_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME',
                                          name='inception' + i + '_avgpool')(x)  # 1,1,1280
    inception9_avgpool_conv = Conv2D(192, 1,
                                     name='inception' + i + '_avgpool_conv')(inception9_avgpool)  # 1,1,192
    inception9_avgpool_bn = BatchNormalization(axis=1,
                                               name='inception' + i + '_avgpool_bn')(inception9_avgpool_conv)
    inception9_avgpool_relu = Activation('relu',
                                         name='inception' + i + '_avgpool_relu')(inception9_avgpool_bn)

    x = concatenate([inception9_1x1relu, inception9_1331concat, inception9_331331concat, inception9_avgpool_relu],
                    axis=-1, name='inception' + i + '_concat')  # 1,1,2048 (320+768+768+192)
inception10_concat = x

# FC
avgpool = GlobalAveragePooling2D(name='avgpool')(inception10_concat)
output_tensor = Dense(200, activation='softmax', name='output')(avgpool)

# Create a model
inceptionv3 = Model(input_tensor, output_tensor, name='inceptionv3')
inceptionv3.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
inceptionv3.compile(loss='sparse_categorical_crossentropy',
                 optimizer=opt,
                 metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
inceptionv3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)  # train_generator, batch_size=batch_size, epochs=epochs)  #

# Test the model with test set
inceptionv3.evaluate(x_test, y_test, verbose=2)  # valid_generator, verbose=2)  #

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
