#####
# Image semantic segmentation

# Dataset: harvard dataverse prostate
# ref. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP
# In the masks, pixel indices correspond to classes as follows:
# 0=Benign (green), 1=Gleason_3 (blue), 2=Gleason_4 (yellow), 3=Gleason_5 (red), 4=unlabelled (white).
# ==> total 5 output

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import os
import tarfile
import random
import math
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate

# Hyper-parameters
rnd_freq = 10  # number of randomly cropped images to generate
rndcrop_size = (216, 216)  # W, H
resize_size = (72, 72)   # W, H
output_size = 5  # output channel size; 5 for harvard dataverse prostate dataset
learning_rate = 0.05  # for u-net, start with larger learning rate
batch_size = 16
epochs = 20

# Results
# ==> input size: 72*72, learning rate: 0.05, batch size: 16, epochs: 20; acc: ~0.03

# Load Harvard dataverse prostate dataset
# 1. Download all 15 tar.gz files from the link below:
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP

# 2. Understand the directory structure
#   + dataverse_files
#       + Gleason_masks_test_pathologist1.tar.gz: test masks 1; png files
#       + Gleason_masks_test_pathologist2.tar.gz: test masks 2; png files
#       + ZT80_38_A~C.tar.gz: as all together, test images; jpg files
#       + Gleason_masks_train.tar.gz: train masks; png files
#       + rest of ZTxx_x_x.tar.gz: as all together, train images; jpg files

# Check the name matching, since some of the images does not have the corresponding masks
train_tar = ['Gleason_masks_train', 'ZT76_39_A', 'ZT76_39_B', 'ZT111_4_A', 'ZT111_4_B', 'ZT111_4_C',
             'ZT199_1_A', 'ZT199_1_B', 'ZT204_6_A', 'ZT204_6_B']
test_tar = ['Gleason_masks_test_pathologist1',  # 'Gleason_masks_test_pathologist2',
            'ZT80_38_A', 'ZT80_38_B', 'ZT80_38_C']

data_dir = 'D:/PycharmProjects/dataverse_files'


def load_images(is_train, rndcrop_size, resize_size):
    """ Load the input images and masks. """
    if is_train:
        tar = train_tar
    else:
        tar = test_tar

    # Get the path to the tar file and the images inside of it (input images and their corresponding mask images)
    mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, tar)

    xs_crop, ys_crop = [], []
    xs_norm, ys_norm = [], []

    for m in range(len(mask_fname)):  # range(30):  #
        tar_idx = tar.index(os.path.dirname(img_fname[m])) - 1  # excl. train_tar[0] which is the mask folder
        x = img_tar[tar_idx].extractfile(img_fname[m])
        y = mask_tar.extractfile(mask_fname[m])

        x = Image.open(x)
        y = Image.open(y)

        # Preprocess image data (random cropping, resizing, normalization)
        for _ in range(rnd_freq):
            x_crop, y_crop, x_norm, y_norm = preprocess_image(x, y, rndcrop_size, resize_size)

            xs_crop.append(x_crop)
            ys_crop.append(y_crop)
            xs_norm.append(x_norm)
            ys_norm.append(y_norm)

    # imgs = show_img(xs_crop, ys_crop, 100, resize_size)  # num of imgs to show should be a multiple of 4
    # imgs.show()

    x_np = np.asarray(xs_norm)
    y_np = np.asarray(ys_norm)

    return x_np, y_np


def get_tar_fname(data_dir, tar):
    """ Get the path to the tar file and the image files inside of it. """
    img_tar, img_fname = [], []
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()[1:]
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            img_fname += i_tar.getnames()[1:]
    mask_fname, img_fname = match_fname(mask_fname, img_fname)
    return mask_tar, img_tar, mask_fname, img_fname


def match_fname(mask_fname, img_fname):
    """ Match the image file name and the mask file name. """
    mask_fname_match, img_fname_match = [], []
    for m in mask_fname:
        m_fname_base = os.path.splitext(os.path.basename(m))[0]
        m_fname_split = m_fname_base.split('_')
        if 'mask' not in m_fname_split[0]:
            # Skip mask file names starting with '.mask'
            continue
        matcher = '_'.join(m_fname_split[1:4]) + '/' + '_'.join(m_fname_split[1:]) + '.jpg'
        for i in img_fname:
            if matcher == i:
                mask_fname_match.append(m)
                img_fname_match.append(i)
    return mask_fname_match, img_fname_match


def preprocess_image(x, y, rndcrop_size, resize_size):
    """ Preprocess the image data (Randomly crop, resize, and normalize). """
    # Randomly crop and increase the sample image size
    # PIL.Image object has a (width, height) tuple of size
    assert x.size[0] >= rndcrop_size[0]  # width
    assert x.size[1] >= rndcrop_size[1]  # height
    assert x.size[0] == y.size[0]
    assert x.size[1] == y.size[1]
    width = random.randint(0, x.size[0] - rndcrop_size[0])
    height = random.randint(0, x.size[1] - rndcrop_size[1])
    x = x.crop((width, height, (width + rndcrop_size[0]), (height + rndcrop_size[1])))
    y = y.crop((width, height, (width + rndcrop_size[0]), (height + rndcrop_size[1])))

    # Resize
    x = x.resize(resize_size)
    y = y.resize(resize_size)

    # Normalize
    x_norm = np.asarray(x, dtype='float32') / 255.0
    y_norm = np.asarray(y, dtype='float32')
    return x, y, x_norm, y_norm


# *** Need to be fixed
def show_img(xs_crop, ys_crop, num_imgs, resize_size):
    """ Show the pairs of images and corresponding masks. """
    assert num_imgs % 4 == 0
    rnd_idx = random.randint(0, len(xs_crop)-(num_imgs / 2))
    xy_show = xs_crop[rnd_idx:rnd_idx+int(num_imgs / 2)] + ys_crop[rnd_idx:rnd_idx+int(num_imgs / 2)]
    nrow = 4
    ncol = math.ceil(num_imgs / nrow)
    imgs = Image.new('RGB', (resize_size[0] * ncol, resize_size[1] * nrow))
    for i in range(len(xy_show)):
        px, py = resize_size[0] * int(i % ncol), resize_size[0] * int(i // ncol) * 2
        imgs.paste(xy_show[i], (px, py))
    return imgs


'''
def get_tar_fname(data_dir, tar):
    """
    """
    img_tar, img_fname = [], {}
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()[1:]
            mask_fname = get_mask_fname_dict(mask_fname)
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            img_fname[tar[t]] = i_tar.getnames()[1:]
    return mask_tar, img_tar, mask_fname, img_fname

def get_mask_fname_dict(mask_fname):
    """
    Convert the list of mask file names into a dictionary with its corresponding image folder name as a key.
    """
    mask_fname_dict = {}
    for m in range(len(mask_fname)):
        m_fname_base = os.path.splitext(os.path.basename(mask_fname[m]))[0]
        m_fname_split = m_fname_base.split('_')
        if 'mask' not in m_fname_split[0]:
            # Skip mask file names starting with '.mask'
            continue
        tar_key = '_'.join(m_fname_split[1:4])
        # Save the mask file names in dictionary
        if tar_key in mask_fname_dict.keys():
            mask_fname_dict[tar_key].append(mask_fname[m])
        else:
            mask_fname_dict[tar_key] = [mask_fname[m]]
    return mask_fname_dict

mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, train_tar)
'''

x_train, y_train = load_images(True, rndcrop_size, resize_size)  # 641 images
x_valid, y_valid = load_images(False, rndcrop_size, resize_size)  # 245 images with pathologist1

# Construct U-Net model
# *** input shape
input_tensor = Input(shape=resize_size + (3,), name='input_tensor')

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
output_tensor = Conv2D(output_size, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_2)

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
