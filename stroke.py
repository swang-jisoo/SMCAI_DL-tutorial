#####
# Image semantic segmentation

# Dataset: stroke dicom files given by park
# 0 = background, 1 = lesion
# ==> total 2 output

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

'''
fpath = './stroke_dcm/input/0102LAADWI0005.dcm'
ds = dcmread(fpath)

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
'''

mask_dir = 'D:/PycharmProjects/stroke_dcm/GT'
img_dir = 'D:/PycharmProjects/stroke_dcm/input'

(_, _, mask_f) = next(os.walk(mask_dir))
(_, _, img_f) = next(os.walk(img_dir))

fname = []
for m in mask_f:
    m_base = os.path.splitext(m)[0]
    for i in img_f:
        i_base = os.path.splitext(i)[0]
        if m_base != i_base:
            continue
        fname.append(m_base)
        #x_np.append(os.path.join(img_dir, i))
        #y_np.append(os.path.join(mask_dir, m))

# AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
# x_np[71] = './stroke_dcm/input\\281SVODWI0001.dcm'
# x_np[224] = './stroke_dcm/input\\286SVODWI0001.dcm'
'''
x_open = [dcmread(x).pixel_array for x in x_np[:71] + x_np[97:224] + x_np[247:]]
y_open = [np.asarray(Image.open(y), dtype='float32') for y in y_np[:71] + y_np[97:224] + y_np[247:]]
x_open = np.asarray(x_open)
y_open = np.asarray(y_open)
'''

rndcrop_size = (72,72)
fname_cut = fname[:71] + fname[97:224] + fname[247:]
x_train, y_train = [], []
for f in fname_cut:
    x = dcmread(os.path.join(img_dir, f + '.dcm')).pixel_array
    y = Image.open(os.path.join(mask_dir, f + '.png'))
    x = zoom(x, rndcrop_size[0] / x.shape[0])
    y = y.resize(rndcrop_size)
    y = np.asarray(y, dtype='float32')

    '''w_start = (x.shape[0] // 2) - (rndcrop_size[0] // 2)
    h_start = (x.shape[1] // 2) - (rndcrop_size[1] // 2)
    x = x[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]
    y = y[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]'''
    x_train.append(x)
    y_train.append(y)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


resize_size = rndcrop_size  # W, H; multiple of 8
output_size = 2
learning_rate = 0.05  # for u-net, start with larger learning rate
batch_size = 16
epochs = 5

# *** input shape
input_tensor = Input(shape=resize_size + (1,), name='input_tensor')

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