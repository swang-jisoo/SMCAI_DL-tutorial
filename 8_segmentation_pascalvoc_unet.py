#####
# Image (semantic/class) segmentation
# It aims to segment the image, i.e. clustering parts of an image together which belong to the same object.
# In other words, semantic segmentation is a image classification at pixel level (thus, localization is important).
# It outputs a pixel-wise mask of the image = labels for each pixel of the image with a category label; e.g.:
#   Class 1 : Pixel belonging to an object. ( 20 objects in Pascal VOC)
#   Class 2 : Pixel bordering the object. (Not applicable in Pascal VOC)
#   Class 3 : None of the above/ Surrounding pixel. (predict total 21 objects in Pascal VOC)
# ref.
# https://www.tensorflow.org/tutorials/images/segmentation
# https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb

# Dataset: Pascal VOC
# description: a set of colored images with annotation files
# images: shape=(, , 3)
# labels: num_classes=20
# (0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow,
# 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tvmonitor,
# 255 =void or unlabelled)
# total_num_examples=2913
# split: training=1465 , val=1450
# ref: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html

# Model: U-net
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

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import os
from PIL import Image
import matplotlib.pyplot as plt
# import mxnet as mx
# from mxnet import image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate

# Hyper-parameters
center_crop_size = (256, 256)
resize_size = (256, 256)
DATA_SHAPE = center_crop_size + (3, )
OUTPUT_SIZE = 21
LEARNING_RATE = 1e-6
BATCH_SIZE = 4
EPOCHS = 10
DIM = 16
KERNEL_SIZE = 3
VERBOSE = 1

# GPU
gpu = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpu[0], True)
except RuntimeError as e:
    print(e)  # Error

# Results
# ==> input size: 72*72, learning rate: 0.05, batch size: 16, epochs: 20; acc: ~0.48

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
voc_dir = 'C:\\Users\\swang\\PycharmProjects' + '/data - pascalvoc/VOC2012'

#   3. Understand the directory structure
#       + VOCdevkit
#           + VOC2012
#               + Annotations: annotations for object detection; xml files
#               + ImageSets: list of image file name classified by classes or train/trainval/val; txt files
#               + JPEGImages: input images; jpg files
#               + SegmentationClass: segmentation label (mask) by class (semantic); png files
#               + SegmentationObject: segmentation label (mask) by object (instance); png files

#   4. get image and label for semantic segmentation
# Converting mxnet.nd.array to numpy.array can cause troubles. Instead, use PIL library.
def read_voc_images(voc_dir, center_crop_size, resize_size, is_train=True):
    """Read all VOC feature and label images."""
    img_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')

    with open(img_fname, 'r') as f:
        images = f.read().split()

    xs, ys = [], []

    for i, fname in enumerate(images):
        x = Image.open(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg'))
        y = Image.open(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'))

        # crop the image by trimming on all four sides and preserving the center of the image
        #x = crop_center(x, center_crop_size)
        #y = crop_center(y, center_crop_size)

        x = x.resize(resize_size)
        y = y.resize(resize_size)

        # Normalize
        x = np.asarray(x, dtype='float32') / 255.0
        y = np.asarray(y, dtype='uint8')

        # Convert border index into background index
        # labels: 0-20 (here, 0 = background + border region)
        y_noborder = y.copy()
        y_noborder[y_noborder == 255] = 0
        #y_wo255 = tf.keras.utils.to_categorical(y_wo255, OUTPUT_SIZE)

        xs.append(x)
        ys.append(y_noborder)

    x_np = np.asarray(xs)
    y_np = np.asarray(ys)

    return x_np, y_np


def crop_center(pil_img, center_crop_size):
    """ Crop the image to the given size by trimming on all four sides and preserving the center of the image. """
    (crop_width, crop_height) = center_crop_size
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


x_train, y_train = read_voc_images(voc_dir, center_crop_size, resize_size, True)
x_valid, y_valid = read_voc_images(voc_dir, center_crop_size, resize_size, False)


# Build U-net
input = Input(DATA_SHAPE, name='input')

# Encoder
# block 1
conv1_1 = Conv2D(DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv1_1')(input)
conv1_2 = Conv2D(DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv1_2')(conv1_1)
max1 = MaxPooling2D(2, padding='same', name='max1')(conv1_2)

# block 2
conv2_1 = Conv2D(2 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv2_1')(max1)
conv2_2 = Conv2D(2 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv2_2')(conv2_1)
max2 = MaxPooling2D(2, padding='same', name='max2')(conv2_2)

# block 3
conv3_1 = Conv2D(4 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv3_1')(max2)
conv3_2 = Conv2D(4 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv3_2')(conv3_1)
max3 = MaxPooling2D(2, padding='same', name='max3')(conv3_2)

# block 4
conv4_1 = Conv2D(8 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv4_1')(max3)
conv4_2 = Conv2D(8 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv4_2')(conv4_1)
max4 = MaxPooling2D(2, padding='same', name='max4')(conv4_2)

# block 5
conv5_1 = Conv2D(16 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv5_1')(max4)
conv5_2 = Conv2D(16 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='conv5_2')(conv5_1)

# Decoder
# block 4
deconv4_1 = Conv2DTranspose(8 * DIM, 2, 2, activation='relu', name='deconv4_1')(conv5_2)
merge4 = tf.concat([deconv4_1, conv4_2], axis=-1, name='merge4')
deconv4_2 = Conv2D(8 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv4_2')(merge4)
deconv4_3 = Conv2D(8 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv4_3')(deconv4_2)

# block 3
deconv3_1 = Conv2DTranspose(4 * DIM, 2, 2, activation='relu', name='deconv3_1')(deconv4_3)
merge3 = tf.concat([deconv3_1, conv3_2], axis=-1, name='merge3')
deconv3_2 = Conv2D(4 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv3_2')(merge3)
deconv3_3 = Conv2D(4 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv3_3')(deconv3_2)

# block 2
deconv2_1 = Conv2DTranspose(2 * DIM, 2, 2, activation='relu', name='deconv2_1')(deconv3_3)
merge2 = tf.concat([deconv2_1, conv2_2], axis=-1, name='merge2')
deconv2_2 = Conv2D(2 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv2_2')(merge2)
deconv2_3 = Conv2D(2 * DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv2_3')(deconv2_2)

# block 1
deconv1_1 = Conv2DTranspose(DIM, 2, 2, activation='relu', name='deconv1_1')(deconv2_3)
merge1 = tf.concat([deconv1_1, conv1_2], axis=-1, name='merge1')
deconv1_2 = Conv2D(DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv1_2')(merge1)
deconv1_3 = Conv2D(DIM, KERNEL_SIZE, activation='relu', padding='same', name='deconv1_3')(deconv1_2)

output = Conv2D(OUTPUT_SIZE, KERNEL_SIZE, activation='softmax', padding='same', name='output')(deconv1_3)

model = Model(input, output, name='u_net')

# Train
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])

hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

# Predict
imgs = model.predict(x_valid)

num_pic = 10
plt.figure(figsize=(20, 4))
for i in range(num_pic):
    ax = plt.subplot(2, num_pic, i+1)
    plt.imshow(x_valid[i])
    ax = plt.subplot(2, num_pic, i+1+num_pic)
    img_arg = np.argmax(imgs[i], axis=-1)
    plt.imshow(img_arg)
plt.show()
