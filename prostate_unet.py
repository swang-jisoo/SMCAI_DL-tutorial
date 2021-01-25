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
resize_size = (72, 72)  # W, H; multiple of 8
output_size = 5  # output channel size; 5 for harvard dataverse prostate dataset
learning_rate = 0.05  # for u-net, start with larger learning rate
batch_size = 16
epochs = 5

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
# Note: pathologist1 and 2 provides multiple different mask images such as
# 'Gleason_masks_test_pathologist1/mask1_ZT80_38_A_7_10.png'
# 'Gleason_masks_test_pathologist2/mask2_ZT80_38_A_7_10.png'
# According to the relevant paper, the two types of test masks are offered for the qualification of inter-pathologist
# variability. The first pathologist annotated the training dataset as well. More info can be found in the paper.

# Check the name matching, since some of the images does not have the corresponding masks
train_tar = ['Gleason_masks_train', 'ZT76_39_A', 'ZT76_39_B', 'ZT111_4_A', 'ZT111_4_B', 'ZT111_4_C',
             'ZT199_1_A', 'ZT199_1_B', 'ZT204_6_A', 'ZT204_6_B']
test_tar_1 = ['Gleason_masks_test_pathologist1',  # 'Gleason_masks_test_pathologist2',
            'ZT80_38_A', 'ZT80_38_B', 'ZT80_38_C']
test_tar_2 = ['Gleason_masks_test_pathologist2',
            'ZT80_38_A', 'ZT80_38_B', 'ZT80_38_C']

data_dir = 'D:/PycharmProjects/dataverse_files'


def load_images(tar, rndcrop_size, resize_size):
    """ Load the input images and masks. """
    '''if is_train:
        tar = train_tar
    else:
        tar = test_tar'''

    # Get the path to the tar file and the images inside of it (input images and their corresponding mask images)
    mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, tar)

    #img_fname.sort()  # img show
    #mask_fname.sort()  # img show

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

    #xy_img = []  # img show
    #xy_img.append(xs_crop)
    #xy_img.append(ys_crop)

    #imgs = show_img(xy_img, 10, 6, resize_size)
    #imgs.show()  # img show

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
    '''assert x.size[0] >= rndcrop_size[0]  # width
    assert x.size[1] >= rndcrop_size[1]  # height
    assert x.size[0] == y.size[0]
    assert x.size[1] == y.size[1]
    width = random.randint(0, x.size[0] - rndcrop_size[0])
    height = random.randint(0, x.size[1] - rndcrop_size[1])
    x = x.crop((width, height, (width + rndcrop_size[0]), (height + rndcrop_size[1])))
    y = y.crop((width, height, (width + rndcrop_size[0]), (height + rndcrop_size[1])))'''

    # Resize
    x = x.resize(resize_size)
    y = y.resize(resize_size)

    # Normalize
    x_norm = np.asarray(x, dtype='float32') / 255.0
    y_norm = np.asarray(y, dtype='float32')
    return x, y, x_norm, y_norm


# *** Need to be fixed
def show_img(img_set, ncol, nrow, resize_size):
    assert nrow % len(img_set) == 0
    num_imgs = ncol * nrow

    img_table = Image.new('RGB', (resize_size[0] * ncol, resize_size[1] * nrow))
    rnd_idx = [random.randint(0, len(img_set[0])) for _ in range(int(num_imgs / len(img_set)))]

    for n in range(num_imgs):
        px, py = int(n % ncol), int(n // ncol)
        i, j = py % len(img_set), px + (5 * (py // len(img_set)))
        img_table.paste(img_set[i][rnd_idx[j]], (resize_size[0] * px, resize_size[0] * py))

    return img_table


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

x_train, y_train = load_images(train_tar, rndcrop_size, resize_size)  # 641 images
x_valid_1, y_valid_1 = load_images(test_tar_1, rndcrop_size, resize_size)  # 245 images with pathologist1
x_valid_2, y_valid_2 = load_images(test_tar_2, rndcrop_size, resize_size)  # 245 images with pathologist2


# Shuffle the dataset
def shuffle_ds(x, y):
    """ Shuffle the train and test datasets (multi-dimensional array) along the first axis.
        Modify the order of samples in the datasets, while their contents and matching sequence remains the same. """
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    return x, y


x_train, y_train = shuffle_ds(x_train, y_train)
x_valid_1, y_valid_1 = shuffle_ds(x_valid_1, y_valid_1)
x_valid_2, y_valid_2 = shuffle_ds(x_valid_2, y_valid_2)

# Construct U-Net model
# *** input shape
input_tensor = Input(shape=resize_size + (3,), name='input_tensor')

# Contracting path
cont1_1 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_1')(input_tensor)  # 570, 570, 64
cont1_2 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_2')(cont1_1)  # 568, 568, 64

cont2_dwn = MaxPooling2D((2, 2), strides=2, name='cont2_dwn')(cont1_2)  # down-sampling; 284, 284, 64; 124
cont2_1 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_1')(cont2_dwn)  # 282, 282, 128
cont2_2 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_2')(cont2_1)  # 280, 280, 128

cont3_dwn = MaxPooling2D((2, 2), strides=2, name='cont3_dwn')(cont2_2)  # down-sampling; 140, 140, 128; 60
cont3_1 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_1')(cont3_dwn)  # 138, 138, 256
cont3_2 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_2')(cont3_1)  # 136, 136, 256

cont4_dwn = MaxPooling2D((2, 2), strides=2, name='cont4_dwn')(cont3_2)  # down-sampling; 68, 68, 256; 28
cont4_1 = Conv2D(512, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_1')(cont4_dwn)  # 66, 66, 256
cont4_2 = Conv2D(512, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_2')(cont4_1)  # 64, 64, 256

# Expansive path
# *** UpSampling2D vs. Conv2DTranspose:
#   ref. https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker

# Apply activation first, then concat
expn2_up = Conv2DTranspose(256, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn2_up')(cont4_2)  # up-sampling; 104, 104, 256
expn2_concat = concatenate([expn2_up, cont3_2], axis=-1, name='expn2_concat')  # 104, 104, 512

expn3_up = Conv2DTranspose(128, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn3_up')(expn2_concat)  # up-sampling; 200, 200, 128
expn3_concat = concatenate([expn3_up, cont2_2], axis=-1, name='expn3_concat')  # 200, 200, 256

expn4_up = Conv2DTranspose(64, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn4_up')(expn3_concat)  # up-sampling; 392, 392, 64
expn4_concat = concatenate([expn4_up, cont1_2], axis=-1, name='expn4_concat')  # 392, 392, 128

# *** channel number
output_tensor = Conv2D(output_size, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_concat)
# output_tensor = Conv2DTranspose(output_size, 1, padding='same')(expn4_concat)

# Create a model
u_net = Model(input_tensor, output_tensor, name='u_net')
u_net.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train, epochs=epochs)

# Test the model with test set
u_net.evaluate(x_valid_1, y_valid_1, verbose=2)
u_net.evaluate(x_valid_2, y_valid_2, verbose=2)

img_1 = u_net.predict(x_valid_1)
img_2 = u_net.predict(x_valid_2)
