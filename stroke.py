#####
# Image semantic segmentation

# Dataset: stroke dicom files given by park
# 0 = background, 1 = lesion

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
from pydicom import dcmread
from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import zoom

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

# Load stroke dataset

# directory structure
#   + stroke_dcm
#       + GT: mask; png files
#       + input: input image; dcm files

mask_dir = 'D:/PycharmProjects/stroke_dcm/GT'
img_dir = 'D:/PycharmProjects/stroke_dcm/input'

(_, _, mask_f) = next(os.walk(mask_dir))
(_, _, img_f) = next(os.walk(img_dir))

mask_f.sort()
img_f.sort()

# filter the file name that exists in both image and mask folders
fname = []
for m in mask_f:
    m_base = os.path.splitext(m)[0]
    for i in img_f:
        i_base = os.path.splitext(i)[0]
        if m_base != i_base:
            continue
        fname.append(m_base)

# some of images are excluded due to the error below:
# AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
# x_np[71] = './stroke_dcm/input\\281SVODWI0001.dcm'
# x_np[224] = './stroke_dcm/input\\286SVODWI0001.dcm'

fname_cut = fname[:71] + fname[97:224] + fname[247:]

'''
# files with lesion segmented mask
fname_cut = ['0102LAADWI0012', '0102LAADWI0013', '0102LAADWI0014', '0102LAADWI0015', '0102LAADWI0016',
             '0102LAADWI0017', '0102LAADWI0018', '0102LAADWI0019', '0102LAADWI0020', '0102LAADWI0021',
             '0102LAADWI0022', '0102LAADWI0023', '0120LAADWI0013', '0120LAADWI0014', '0120LAADWI0015',
             '0120LAADWI0016', '280SVODWI0003', '280SVODWI0014', '281SVODWI0018', '282SVODWI0006',
             '283SVODWI0017', '283SVODWI0018', '284SVODWI0007', ... etc]
'''

rndcrop_size = (96, 96)
x_all, y_all = [], []
for f in fname_cut:
    x = dcmread(os.path.join(img_dir, f + '.dcm')).pixel_array
    x = zoom(x, rndcrop_size[0] / x.shape[0])
    x = x.astype('float32')/2048.0

    y = Image.open(os.path.join(mask_dir, f + '.png')).convert('L')  # from rgb to greyscale
    #y = y.point(lambda p: p > 0)  # image thresholding; **** order of thresholding and resize
    y = y.resize(rndcrop_size, resample = Image.BICUBIC)  # default resample = PIL.Image.BICUBIC
    y = y.point(lambda p: p > 0.5)  # 0 -> 0.5 due to BICUBIC
    y = np.asarray(y, dtype='float32')

    # collect the masks where lesions are segmented
    if y.max() == 0.:
        continue

    '''w_start = (x.shape[0] // 2) - (rndcrop_size[0] // 2)
    h_start = (x.shape[1] // 2) - (rndcrop_size[1] // 2)
    x = x[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]
    y = y[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]'''
    x_all.append(x)
    y_all.append(y)

x_all = np.asarray(x_all)
y_all = np.asarray(y_all)


# Shuffle the dataset
def shuffle_ds(x, y):
    """ Shuffle the train and test datasets (multi-dimensional array) along the first axis.
        Modify the order of samples in the datasets, while their contents and matching sequence remains the same. """
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    return x, y


# set the dataset of 2 patients (last 52 images) as a testset
#x_train, y_train = shuffle_ds(x_all[:-52], y_all[:-52])
#x_valid, y_valid = x_all[-52:], y_all[-52:]

# filtered dataset (only incl. lesion segmented images)
x_train, y_train = shuffle_ds(x_all[:-5], y_all[:-5])
x_valid, y_valid = x_all[-5:], y_all[-5:]

resize_size = rndcrop_size  # W, H; multiple of 8
output_size = 1  # binary segmentation
learning_rate = 0.00001  # for u-net, start with larger learning rate
batch_size = 2
epochs = 150

# *** input shape
input_tensor = Input(shape=resize_size + (1,), name='input_tensor')

# Contracting path
cont1_1 = Conv2D(32, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_1')(input_tensor)  # 570, 570, 64
cont1_2 = Conv2D(32, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_2')(cont1_1)  # 568, 568, 64

cont2_dwn = MaxPooling2D((2, 2), strides=2, name='cont2_dwn')(cont1_2)  # down-sampling; 284, 284, 64; 124
cont2_1 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_1')(cont2_dwn)  # 282, 282, 128
cont2_2 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_2')(cont2_1)  # 280, 280, 128

cont3_dwn = MaxPooling2D((2, 2), strides=2, name='cont3_dwn')(cont2_2)  # down-sampling; 140, 140, 128; 60
cont3_1 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_1')(cont3_dwn)  # 138, 138, 256
cont3_2 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_2')(cont3_1)  # 136, 136, 256

cont4_dwn = MaxPooling2D((2, 2), strides=2, name='cont4_dwn')(cont3_2)  # down-sampling; 68, 68, 256; 28
cont4_1 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_1')(cont4_dwn)  # 66, 66, 256
cont4_2 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_2')(cont4_1)  # 64, 64, 256

# Expansive path
# *** UpSampling2D vs. Conv2DTranspose:
#   ref. https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker

# Apply activation first, then concat
expn2_up = Conv2DTranspose(128, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn2_up')(cont4_2)  # up-sampling; 104, 104, 256
expn2_concat = concatenate([expn2_up, cont3_2], axis=-1, name='expn2_concat')  # 104, 104, 512

expn3_up = Conv2DTranspose(64, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn3_up')(expn2_concat)  # up-sampling; 200, 200, 128
expn3_concat = concatenate([expn3_up, cont2_2], axis=-1, name='expn3_concat')  # 200, 200, 256

expn4_up = Conv2DTranspose(32, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn4_up')(expn3_concat)  # up-sampling; 392, 392, 64
expn4_concat = concatenate([expn4_up, cont1_2], axis=-1, name='expn4_concat')  # 392, 392, 128

# *** channel number
output_tensor = Conv2D(output_size, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_concat)  # sigmoid -> softmax [0-1] (binary)
# output_tensor = Conv2DTranspose(output_size, 1, padding='same')(expn4_concat)

# Create a model
u_net = Model(input_tensor, output_tensor, name='u_net')
u_net.summary()


# Dice Coefficient to work with Tensorflow
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    #tf.print(numerator, denominator)
    return tf.reduce_mean(numerator / (denominator+1))

def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss=dice_loss, optimizer=opt, metrics=[dice_score])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)  #

# Test the model with test set
u_net.evaluate(x_valid, y_valid, verbose=2)

img = u_net.predict(x_valid)
#img_arg = np.argmax(img, axis=-1)
#img_arg = img_arg[..., tf.newaxis]
#img_arg = img_arg.astype('float32')
img_arg = img * 255
A = img[1]

img_set = []
img_set.append(x_valid)
img_set.append(y_valid)
img_set.append(img)

columns = 5
rows = 3


def show_img(img_set, ncol, nrow):
    assert nrow % len(img_set) == 0
    num_imgs = ncol * nrow

    fig = plt.figure(figsize=(8, 8))
    #rnd_idx = [random.randint(0, len(img_set[0])) for _ in range(int(num_imgs / len(img_set)))]
    rnd_idx = 0

    for n in range(num_imgs):
        fig.add_subplot(rows, columns, n+1)
        px, py = int(n % ncol), int(n // ncol)
        i, j = py % len(img_set), px + (ncol * (py // len(img_set)))
        plt.imshow(img_set[i][j])

    return plt.show()


show_img(img_set, columns, rows)

'''
x = dcmread('./stroke_dcm/input/0102LAADWI0018.dcm').pixel_array
x = zoom(x, rndcrop_size[0] / x.shape[0])
x_t = np.array([x])

y = Image.open('./stroke_dcm/GT/0102LAADWI0018.png').convert('L')
y = y.resize(rndcrop_size)
y = y.point(lambda p: p > 0)

img = u_net.predict(x_t)
img_arg = np.argmax(img, axis=-1)[0]

plt.imshow(x)
plt.imshow(y)
plt.imshow(img_arg)
'''

'''
fpath = './stroke_dcm/input/0102LAADWI0005.dcm'
ds = dcmread(fpath)

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
'''
