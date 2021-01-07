#####
# Image (semantic) segmentation
# For each pixel in a test image, predict the class of the object containing that pixel or
# `background' if the pixel does not belong to one of the 20 specified classes.

# Dataset: Pascal VOC
# A dataset that can be used for classification, segmentation, detection, and action classification.
# There are 20 object classes.

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
#   Class 1 : Pixel belonging to an object. ( 20 objects in Pascal VOC)
#   Class 2 : Pixel bordering the object. (Not applicable in Pascal VOC)
#   Class 3 : None of the above/ Surrounding pixel. (predict total 21 objects in Pascal VOC)

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
#   - Weighted loss: designed to tackle imbalanced data in back/foreground classification;
#     up-weight the mis-classification error of less frequent classes in the cross-entropy loss

# Import necessary libraries
import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate
from PIL import Image
import numpy as np

# Hyper-parameters
ori_size = (216, 216)
img_size = (72, 72)
learning_rate = 0.05
epochs = 20

# Results
# ==>

# Load the Pascal VOC dataset
#   1. Download the Pascal VOC dataset and unzip the tar file:
'''
# download the Pascal VOC dataset (cmd)
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# unzip the tar file (1) (cmd)
tar xf VOCtrainval_11-May-2012.tar
# OR (2) (python)
import tarfile
import mxnet  # may need to install the package (run on cmd: pip install mxnet)

base_dir = 'C:/Users/SMC/PycharmProjects'  # this may differ from your folder location
tar_dir = base_dir + '/VOCtrainval_11-May-2012.tar'
fp = tarfile.open(tar_dir, 'r')
fp.extractall(base_dir)  # extract the tar file
'''

#   2. Get the path to the folder where the dataset exists
voc_dir = 'D:/PycharmProjects' + '/VOCdevkit/VOC2012'


#   3. Understand the directory structure
#       + VOCdevkit
#           + VOC2012
#               + Annotations: annotations for object detection; xml files
#               + ImageSets: list of image file name classified by classes or train/trainval/val; txt files
#               + JPEGImages: input images; jpg files
#               + SegmentationClass: segmentation label (mask) by class (semantic); png files
#               + SegmentationObject: segmentation label (mask) by object (instance); png files

#   4. get image and label for semantic segmentation
def read_voc_images(voc_dir, img_size, is_train=True):
    """Read all VOC feature and label images."""
    img_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')

    with open(img_fname, 'r') as f:
        images = f.read().split()

    xs, ys = [], []
    for i, fname in enumerate(images):
        x = Image.open(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg'))
        y = Image.open(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'))

        # preprocessing
        # crop the image by trimming on all four sides and preserving the center of the image
        # normalize
        width, height = x.size
        cx = int(width/2)
        cy = int(height/2)

        roi = (cx-ori_size[0]/2, cy-ori_size[1]/2, cx+ori_size[0]/2, cy+ori_size[1]/2)

        x = x.crop(roi)
        y = y.crop(roi)

        x = x.resize(img_size)
        y = y.resize(img_size)

        x = np.asarray(x)/255.0
        y = np.asarray(y)/255.0

        xs.append(x)
        ys.append(y)

    return xs, ys


x_train, y_train = read_voc_images(voc_dir, img_size, True)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid, y_valid = read_voc_images(voc_dir, img_size, False)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

# Construct U-Net model
# *** input shape
input_tensor = Input(shape=img_size + (3,), name='input_tensor')

# Contracting path
cont1_1 = Conv2D(64, 3, activation='relu', padding='same', name='cont1_1')(input_tensor)  # 570, 570, 64
cont1_2 = Conv2D(64, 3, activation='relu', padding='same', name='cont1_2')(cont1_1)  # 568, 568, 64

cont2_dwn = MaxPooling2D((2, 2), strides=2, name='cont2_dwn')(cont1_2)  # down-sampling; 284, 284, 64; 124
cont2_1 = Conv2D(128, 3, activation='relu', padding='same', name='cont2_1')(cont2_dwn)  # 282, 282, 128
cont2_2 = Conv2D(128, 3, activation='relu', padding='same', name='cont2_2')(cont2_1)  # 280, 280, 128

cont3_dwn = MaxPooling2D((2, 2), strides=2, name='cont3_dwn')(cont2_2)  # down-sampling; 140, 140, 128; 60
cont3_1 = Conv2D(256, 3, activation='relu', padding='same', name='cont3_1')(cont3_dwn)  # 138, 138, 256
cont3_2 = Conv2D(256, 3, activation='relu', padding='same', name='cont3_2')(cont3_1)  # 136, 136, 256

cont4_dwn = MaxPooling2D((2, 2), strides=2, name='cont4_dwn')(cont3_2)  # down-sampling; 68, 68, 256; 28
cont4_1 = Conv2D(512, 3, activation='relu', padding='same', name='cont4_1')(cont4_dwn)  # 66, 66, 256
cont4_2 = Conv2D(512, 3, activation='relu', padding='same', name='cont4_2')(cont4_1)  # 64, 64, 256

# Expansive path
# *** UpSampling2D vs. Conv2DTranspose:
#   ref. https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker


expn2_up = Conv2DTranspose(256, 2, strides=2, name='expn2_up')(cont4_2)  # up-sampling; 104, 104, 256
cropping_size = (cont3_2.shape[1] - expn2_up.shape[1]) // 2
cropping = ((cropping_size, cropping_size), (cropping_size, cropping_size))
expn2_crop = Cropping2D(cropping, name='expn2_crop')(cont3_2)  # 104, 104, 256
expn2_concat = concatenate([expn2_up, expn2_crop], axis=-1, name='expn2_concat')  # 104, 104, 512
expn2_1 = Conv2D(256, 3, activation='relu', padding='same', name='expn2_1')(expn2_concat)  # 102, 102, 256
expn2_2 = Conv2D(256, 3, activation='relu', padding='same', name='expn2_2')(expn2_1)  # 100, 100, 256

expn3_up = Conv2DTranspose(128, 2, strides=2, name='expn3_up')(expn2_2)  # up-sampling; 200, 200, 128
cropping_size = (cont2_2.shape[1] - expn3_up.shape[1]) // 2
cropping = ((cropping_size, cropping_size), (cropping_size, cropping_size))
expn3_crop = Cropping2D(cropping, name='expn3_crop')(cont2_2)  # 200, 200, 128
expn3_concat = concatenate([expn3_up, expn3_crop], axis=-1, name='expn3_concat')  # 200, 200, 256
expn3_1 = Conv2D(128, 3, activation='relu', padding='same', name='expn3_1')(expn3_concat)  # 198, 198, 128
expn3_2 = Conv2D(128, 3, activation='relu', padding='same', name='expn3_2')(expn3_1)  # 196, 196, 128

expn4_up = Conv2DTranspose(64, 2, strides=2, name='expn4_up')(expn3_2)  # up-sampling; 392, 392, 64
cropping_size = (cont1_2.shape[1] - expn4_up.shape[1]) // 2
cropping = ((cropping_size, cropping_size), (cropping_size, cropping_size))
expn4_crop = Cropping2D(cropping, name='expn4_crop')(cont1_2)  # 392, 392, 64
expn4_concat = concatenate([expn4_up, expn4_crop], axis=-1, name='expn4_concat')  # 392, 392, 128
expn4_1 = Conv2D(64, 3, activation='relu', padding='same', name='expn4_1')(expn4_concat)  # 390, 390, 64
expn4_2 = Conv2D(64, 3, activation='relu', padding='same', name='expn4_2')(expn4_1)  # 388, 388, 64

# *** channel number
output_tensor = Conv2D(20 + 1, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_2)

# Create a model
u_net = Model(input_tensor, output_tensor, name='u_net')
u_net.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train, epochs=epochs)

# Test the model with test set
u_net.evaluate(x_valid, y_valid, verbose=2)
