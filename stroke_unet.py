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
import matplotlib.pyplot as plt
from pydicom import dcmread
from PIL import Image
from scipy.ndimage.interpolation import zoom
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate


# parameters
rndcrop_size = (96, 96)
resize_size = rndcrop_size
output_size = 1  # binary segmentation
learning_rate = 0.00001
batch_size = 2
epochs = 150

# Results
# ==> input size: 96*96, learning rate: 0.00001, batch size: 2, epochs: 150; dice: 0.3306
# Memory allocation: rescale the image size and reduce the depth of the network
# Imbalance: 90% of images in the dataset don't have a mask with clearly delineated lesions --> resample the datasets
# Imbalance: Lesions take up a small portion of the entire image --> change accuracy/crossentropy to dice score/loss
# Convergence optimization: failed to converge --> lower learning rate & batch size
#

# Define necessary functions
def match_fname(mask_dir, img_dir):
    """ Return the list of file names that exist in both image and mask folders """
    # Get file names in the folder
    (_, _, mask_f) = next(os.walk(mask_dir))
    (_, _, img_f) = next(os.walk(img_dir))

    mask_f.sort()
    img_f.sort()

    # Filter the file name that exists in both image and mask folders
    fname = []
    for m in mask_f:
        m_base = os.path.splitext(m)[0]
        for i in img_f:
            i_base = os.path.splitext(i)[0]
            if m_base != i_base:
                continue
            fname.append(m_base)

    return fname


def load_images(fname, rndcrop_size):
    """ Load the numpy array of preprocessed image and mask datasets """
    xs, ys = [], []
    for f in fname:
        # Rescale the input images
        x = dcmread(os.path.join(img_dir, f + '.dcm')).pixel_array
        x = zoom(x, rndcrop_size[0] / x.shape[0])
        x = x.astype('float32') / 2048.0  # normalization

        # Rescale the mask images
        y = Image.open(os.path.join(mask_dir, f + '.png')).convert('L')  # from rgb to greyscale
        y = y.resize(rndcrop_size, resample=Image.BICUBIC)  # default resample = PIL.Image.BICUBIC
        # Make an index for the part of lesions as 1
        # (image thresholding) resize the image first and then apple thresholding
        y = y.point(lambda p: p > 0.5)  # *** 0.5 due to BICUBIC
        y = np.asarray(y, dtype='float32')

        # Collect only the masks where lesions are clearly delineated
        if y.max() == 0.:
            continue

        '''
        # Center crop the image
        w_start = (x.shape[0] // 2) - (rndcrop_size[0] // 2)
        h_start = (x.shape[1] // 2) - (rndcrop_size[1] // 2)
        x = x[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]
        y = y[w_start:w_start + rndcrop_size[0], h_start:h_start + rndcrop_size[1]]
        '''

        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return xs, ys


def shuffle_ds(x, y):
    """ Shuffle the train and test datasets (multi-dimensional array) along the first axis.
        Modify the order of samples in the datasets, while their contents and matching sequence remains the same. """
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]

    return x, y


# Dice score and loss function
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # tf.print(numerator, denominator)
    return tf.reduce_mean(numerator / (denominator+1))


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


def show_img(img_set, ncol, nrow):
    """ Plot the list of images consisting of input image, mask, and predicted result """
    assert nrow % len(img_set) == 0
    num_imgs = ncol * nrow

    fig = plt.figure(figsize=(8, 8))
    # rnd_idx = [random.randint(0, len(img_set[0])) for _ in range(int(num_imgs / len(img_set)))]

    for n in range(num_imgs):
        fig.add_subplot(nrow, ncol, n+1)
        px, py = int(n % ncol), int(n // ncol)
        i, j = py % len(img_set), px + (ncol * (py // len(img_set)))
        plt.imshow(img_set[i][j])

    return plt.show()


# Load the dataset
# directory structure
#   + stroke_dcm
#       + GT: mask; png files
#       + input: input image; dcm files

mask_dir = 'D:/PycharmProjects/stroke_dcm/GT'
img_dir = 'D:/PycharmProjects/stroke_dcm/input'

# Get the file names to read
fname = match_fname(mask_dir, img_dir)
fname_cut = fname[:71] + fname[97:224] + fname[247:]
# Note: some of images are excluded due to the error below:
# AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
# e.g.
# x_np[71] = './stroke_dcm/input\\281SVODWI0001.dcm'
# x_np[224] = './stroke_dcm/input\\286SVODWI0001.dcm'
'''
# files with lesion segmented mask
fname_cut = ['0102LAADWI0012', '0102LAADWI0013', '0102LAADWI0014', '0102LAADWI0015', '0102LAADWI0016',
             '0102LAADWI0017', '0102LAADWI0018', '0102LAADWI0019', '0102LAADWI0020', '0102LAADWI0021',
             '0102LAADWI0022', '0102LAADWI0023', '0120LAADWI0013', '0120LAADWI0014', '0120LAADWI0015',
             '0120LAADWI0016', '280SVODWI0003', '280SVODWI0014', '281SVODWI0018', '282SVODWI0006',
             '283SVODWI0017', '283SVODWI0018', '284SVODWI0007', ... etc]
'''

# Load the preprocessed image and mask datasets
x_all, y_all = load_images(fname_cut, rndcrop_size)

# Shuffle the dataset and split into train and test sets
# x_train, y_train = shuffle_ds(x_all[:-52], y_all[:-52])
# x_valid, y_valid = x_all[-52:], y_all[-52:]  # (last 2 patients)

# Shuffle the dataset and split train and test sets
# (the entire dataset only includes the images with clearly delineated lesions on the mask)
x_train, y_train = shuffle_ds(x_all[:-5], y_all[:-5])
x_valid, y_valid = x_all[-5:], y_all[-5:]


# Construct U-Net model
input_tensor = Input(shape=resize_size + (1,), name='input_tensor')

# Contracting path
cont1_1 = Conv2D(32, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_1')(input_tensor)
cont1_2 = Conv2D(32, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont1_2')(cont1_1)

cont2_dwn = MaxPooling2D((2, 2), strides=2, name='cont2_dwn')(cont1_2)
cont2_1 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_1')(cont2_dwn)
cont2_2 = Conv2D(64, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont2_2')(cont2_1)

cont3_dwn = MaxPooling2D((2, 2), strides=2, name='cont3_dwn')(cont2_2)
cont3_1 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_1')(cont3_dwn)
cont3_2 = Conv2D(128, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont3_2')(cont3_1)

cont4_dwn = MaxPooling2D((2, 2), strides=2, name='cont4_dwn')(cont3_2)
cont4_1 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_1')(cont4_dwn)
cont4_2 = Conv2D(256, 3, padding='same',
                 activation='relu', kernel_initializer='he_normal', name='cont4_2')(cont4_1)

# Expansive path
expn2_up = Conv2DTranspose(128, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn2_up')(cont4_2)
expn2_concat = concatenate([expn2_up, cont3_2], axis=-1, name='expn2_concat')

expn3_up = Conv2DTranspose(64, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn3_up')(expn2_concat)
expn3_concat = concatenate([expn3_up, cont2_2], axis=-1, name='expn3_concat')

expn4_up = Conv2DTranspose(32, 2, strides=2, padding='same',
                           activation='relu', kernel_initializer='he_normal', name='expn4_up')(expn3_concat)
expn4_concat = concatenate([expn4_up, cont1_2], axis=-1, name='expn4_concat')

# *** sigmoid vs softmax for binary and multi-class segmentation
output_tensor = Conv2D(output_size, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_concat)

# Create a model
u_net = Model(input_tensor, output_tensor, name='u_net')
u_net.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss=dice_loss, optimizer=opt, metrics=[dice_score])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Test the model with test set
u_net.evaluate(x_valid, y_valid, verbose=2)

# Generate the predicted result and plot it with the original image and mask
img = u_net.predict(x_valid)
'''
img_arg = np.argmax(img, axis=-1)
img_arg = img_arg[..., tf.newaxis]
img_arg = img_arg.astype('float32')
img_arg = img * 255
'''

img_set = [x_valid, y_valid, img]
ncol, nrow = 5, 3
show_img(img_set, ncol, nrow)

'''
fpath = './stroke_dcm/input/0102LAADWI0005.dcm'
ds = dcmread(fpath)

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
'''
