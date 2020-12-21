#####
# ref.
# https://www.tensorflow.org/tutorials/images/segmentation
# https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb
#####

# Image (semantic) segmentation
# It aims to segment the image, e.g. the location, shape of objects in the image.
# Thus, it outputs a pixel-wise mask of the image = labels for each pixel of the image with a category label:
#   Class 1 : Pixel belonging to an object.
#   Class 2 : Pixel bordering the object.
#   Class 3 : None of the above/ Surrounding pixel.

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

# Load the Oxford-IIIT Pet Dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
