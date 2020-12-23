#####
# Image (semantic) segmentation
# Dataset: Pascal VOC
# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Image (semantic) segmentation
# ref.
# https://www.tensorflow.org/tutorials/images/segmentation
# https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb
# It aims to segment the image, i.e. clustering parts of an image together which belong to the same object.
# In other words, semantic segmentation is a image classification at pixel level (thus, localization is important).
# It outputs a pixel-wise mask of the image = labels for each pixel of the image with a category label; e.g.:
#   Class 1 : Pixel belonging to an object.
#   Class 2 : Pixel bordering the object.
#   Class 3 : None of the above/ Surrounding pixel.

# Fully Convolution Network (FCN)

# U-Net
# ref. U-Net: Convolutional Networks for Biomedical Image Segmentation
# Architecture
#   (1) Contracting path (left side of U-shape; encoder; typical convolutional network)
#       - Max-pooling: increase the number of feature channels to propagate
#       - Down-sampling: take high resolution features (context information) from the image, which are cropped and
#         combined with up-sampling output to keep accurate localization information
#   (2) Expansive path (right side of U-shape; decoder; symmetric to the contracting path)
#       - Up-sampling: increase the resolution of the output, combined with the cropped high resolution feature
#         from the contrasting path to localize
#   (3) (no FC layers): the segmentation map only contains the pixel
# Other techniques used
#   - Data augmentation: a technique to increase the diversity of the training set by applying transformations such
#     as image rotation; here, we use elastic deformation to efficiently train with very few annotated images
#       - Elastic deformation: a change in shape of a material at low stress that is recoverable after the stress
#         is removed; sort of skewing the pixel value
#         ref. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis
#         *** operation?
#   - Overlap tile: fill the missing context of the border region with the mirrored input image.
#   - Weighted loss:

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, concatenate

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

# Hyper-parameters

# Results
# ==>

# Load the Pascal VOC dataset

# Construct U-Net model
x = Input(shape=(572, 572, 1))
# Contracting path
x = Conv2D(64, 3, activation='relu', name='')(x)
x = Conv2D(64, 3, activation='relu', name='')(x)
x = MaxPooling2D((2, 2), strides=2, padding='valid', name='')(x)  # down-sampling

# Expansive path
