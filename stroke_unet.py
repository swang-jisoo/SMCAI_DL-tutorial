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
import re
import matplotlib.pyplot as plt
from pydicom import dcmread
from PIL import Image
from scipy.ndimage.interpolation import zoom
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate


# parameters
is_DWI_only = True  # DWI only if True else ADC + DWI concat
is_subtype = True  # specific subtype will be chosen if True else all combined types
subtypes = ['LAA', 'CE', 'SVO']  # subtypes[subtype_idx]
subtype_idx = 1  # specify the index if is_subtype == True
is_tiff = True  # input data will be tiff format if True else dicom format

root_dir = 'C:/Users/SMC/Dropbox/ESUS_ML'
DCMv_dir = ['DCM_gtmaker_v2_release', 'DCM_gtmaker_v3', 'DCM_gtmaker_v5']

# for DWI only, last 5 images belong to one patient who are excluded in the train set (DCM_gtmaker_v5\GT\299SVODWI0013 ~ 17)
# for DWI only, last 3, 11, 5 images belong to laa, ce, svo patients respectively (laa: v5\GT\0120, ce: v3\GT\164, v5: v5\GT\299)
# for ADC+DWI, last 11 images (DCM_gtmaker_v3\GT\164CEDWI0011 ~ 21)
# for ADC+DWI, it contains ce patients only
# test_idx = 5 if is_DWI_only else 11
# test_idx = {True: {True: [3, 11, 5], False: 5}, False: {True: [0, 11, 0], False: 11}}  # test_idx[is_DWI_only][is_subtype]

max_dim = 256
depth = 4

rndcrop_size = (96, 96)
resize_size = rndcrop_size
learning_rate = 0.0005
batch_size = 2
epochs = 50
output_size = 1  # binary segmentation

# Results
# ==> data: v5 DWI 53, input size: 96*96, learning rate: 0.00001, batch size: 2, epochs: 150; dice: 0.26 ~ 0.33
# Memory allocation: rescale the image size and reduce the depth of the network
# Imbalance: 90% of images in the dataset don't have a mask with clearly delineated lesions --> resample the datasets
# Imbalance: Lesions take up a small portion of the entire image --> change accuracy/crossentropy to dice score/loss
# Convergence optimization: failed to converge --> lower learning rate & batch size, higher epochs

# ==> data: all DWI 98, input size: 96*96, learning rate: 0.00001, batch size: 2, epochs: 150; dice: ~ 0.12
# ==> data: all DWI, input size: 96*96, learning rate: 0.00005, batch size: 2, epochs: 150; dice: 0.51
# ==> data: all DWI, input size: 96*96, learning rate: 0.0001, batch size: 2, epochs: 150; dice: 0.66
# ==> data: all DWI, input size: 96*96, learning rate: 0.0005, batch size: 3, epochs: 150; dice: ~ 0.75
# ==> data: all DWI, input size: 96*96, learning rate: 0.0005, batch size: 4, epochs: 150; dice: 0.65
# ==> data: all DWI, input size: 96*96, learning rate: 0.0005, batch size: 3, epochs: 200; dice: 0.70

# ==> data: ADC+DWI 45, input size: 96*96, learning rate: 0.0005, batch size: 3, epochs: 150; dice: ~0.85


# Define necessary functions
def get_matched_fpath(is_DWI_only, folder_dir):
    """ Return a list of path of input images containing 'DWI' in the 'input' folder under the given directory. """
    # Get file names in the folder
    mask_dir = os.path.join(folder_dir, 'GT')
    img_dir = os.path.join(folder_dir, 'DCM' if is_tiff else 'input')
    (_, _, mask_f) = next(os.walk(mask_dir))
    (_, _, img_f) = next(os.walk(img_dir))

    mask_f.sort()
    img_f.sort()

    fname_dwi = []
    for i in img_f:
        if 'DWI' in i:
            f_dwi = os.path.splitext(i)[0]
            f_adc = f_dwi[:f_dwi.index('DWI')] + 'ADC' + f_dwi[f_dwi.index('DWI') + 3:]
            if is_DWI_only and (f_dwi + '.png' in mask_f):
                # returns file names containing 'DWI', which has a mask
                fname_dwi.append(os.path.join(img_dir, i))
            elif (not is_DWI_only) and (f_adc + '.dcm' in img_f) and (f_dwi + '.png' in mask_f):
                # returns file names containing 'DWI', which has a corresponding name containing 'ADC' and has a mask
                fname_dwi.append(os.path.join(img_dir, i))

    return fname_dwi  # {folder_dir: fname_dwi}


def separate_subtypes(fname_dwi):
    fname_sub = []
    for f_dwi in fname_dwi:
        f_base = os.path.basename(f_dwi)
        if subtypes[subtype_idx] in f_base:
            fname_sub.append(f_dwi)
    return fname_sub


def load_images(is_DWI_only, fname_dwi, rndcrop_size):
    """ Load the numpy array of preprocessed image and mask datasets """
    fname_pass, xs, ys = [], [], []
    for f in fname_dwi:
        # Load and rescale the mask images
        f_mask = os.path.join(os.path.dirname(os.path.dirname(f)), 'GT',
                              os.path.splitext(os.path.basename(f))[0] + '.png')
        y = Image.open(f_mask).convert('L')  # from rgb to greyscale
        y = y.resize(rndcrop_size, resample=Image.BICUBIC)  # default resample = PIL.Image.BICUBIC
        # Make an index for the part of lesions as 1
        # (image thresholding) resize the image first and then apple thresholding
        y = y.point(lambda p: p > 0.5)  # *** 0.5 due to BICUBIC
        y = np.asarray(y, dtype='float32')

        # Collect only the masks where lesions are clearly delineated
        if y.max() == 0.:
            continue
        fname_pass.append(f)

        # Load and rescale the input images
        if is_DWI_only:
            if is_tiff:
                x = Image.open(f)
                x = x.resize(rndcrop_size, resample=Image.BICUBIC)
                x = np.array(x, dtype='float32')
                # *** normalization
            else:
                x = dcmread(f).pixel_array
                x = zoom(x, rndcrop_size[0] / x.shape[0])  # rescale
                x = x.astype('float32') / 2048.0  # normalization
        else:
            f_base = os.path.basename(f)
            f_adc = os.path.join(os.path.dirname(f),
                                 f_base[:f_base.index('DWI')] + 'ADC' + f_base[f_base.index('DWI') + 3:])
            if is_tiff:
                x_adc = Image.open(f_adc)
                x_adc = x_adc.resize(rndcrop_size, resample=Image.BICUBIC)
                x_adc = np.array(x_adc, dtype='float32')
                # *** normalization

                x_dwi = Image.open(f)
                x_dwi = x_dwi.resize(rndcrop_size, resample=Image.BICUBIC)
                x_dwi = np.array(x_dwi, dtype='float32')
                # *** normalization
            else:
                x_adc = dcmread(f_adc).pixel_array
                x_adc = zoom(x_adc, rndcrop_size[0] / x_adc.shape[0])  # rescale
                x_adc = x_adc.astype('float32') / 2048.0  # normalization

                x_dwi = dcmread(f).pixel_array
                x_dwi = zoom(x_dwi, rndcrop_size[0] / x_dwi.shape[0])  # rescale
                x_dwi = x_dwi.astype('float32') / 2048.0  # normalization

            # Concatenate ADC and DWI image
            x = np.concatenate((x_adc[:, :, np.newaxis], x_dwi[:, :, np.newaxis]), axis=-1)

        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    test_idx = 0
    pt_idx_last = re.findall('^\d+', os.path.basename(fname_pass[-1]))[0]
    for i in range(len(fname_pass) - 2, 0, -1):
        pt_idx = re.findall('^\d+', os.path.basename(f))[0]
        if pt_idx != pt_idx_last:
            break
        test_idx = i

    return test_idx, xs, ys


def shuffle_ds(x, y):
    """ Shuffle the train and test datasets (multi-dimensional array) along the first axis.
        Modify the order of samples in the datasets, while their contents and matching sequence remains the same. """
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]

    return x, y


def show_img(is_DWI_only, img_set, ncol, nrow):
    """ Plot the list of images consisting of input image, mask, and predicted result """
    fig = plt.figure(figsize=(8, 8))
    num_imgs = ncol * nrow
    ylabels = ['DWI ', 'Ground truth', 'Segmentation'] if is_DWI_only else ['ADC', 'DWI ', 'Ground truth', 'Segmentation']
    # rnd_idx = [random.randint(0, len(img_set[0])) for _ in range(int(num_imgs / len(img_set)))]

    for n in range(num_imgs):
        fig.add_subplot(nrow, ncol, n + 1)
        j, i = int(n % ncol), int(n // ncol)
        if is_DWI_only:
            assert nrow % len(img_set) == 0
            plt.imshow(img_set[i][j])
            plt.ylabel(ylabels[i]) if j == 0 else None
            plt.xticks([]); plt.yticks([])
            # print('img_set[', i, '][', j, ']')
        else:
            assert nrow % (len(img_set) + 1) == 0
            if i == 0:
                plt.imshow(img_set[0][j][:, :, i])
                plt.ylabel(ylabels[i]) if j == 0 else None
                plt.xticks([]); plt.yticks([])
                # print('img_set[0][', j, '][:,:,', i, ']')
            elif i == 1:
                plt.imshow(img_set[0][j][:, :, i])
                plt.ylabel(ylabels[i]) if j == 0 else None
                plt.xticks([]); plt.yticks([])
                # print('img_set[0][', j, '][:,:,', i, ']')
            elif i > 1:
                plt.imshow(img_set[i-1][j])
                plt.ylabel(ylabels[i]) if j == 0 else None
                plt.xticks([]); plt.yticks([])
                # print('img_set[', i, '][', j, ']')

    data = ('TIFF ' if is_tiff else 'DICOM ') + \
           ('Data: DWI_' if is_DWI_only else 'Data: ADC+DWI_') + \
           (subtypes[subtype_idx] if is_subtype else None)
    num_data = '(' + str(len(x_train)) + ' in train, ' + str(len(x_valid)) + ' in test) \n'
    network = 'max dim: ' + str(max_dim) + ', depth: ' + str(depth) + '\n'
    param = 'Parameters: ' + str(rndcrop_size) + ', ' + str(learning_rate) + ', ' + str(batch_size) + ', ' + str(epochs) + '\n'
    metrics = 'Test dice loss & score: ' + str(round(test_result[0], 4)) + ', ' + str(round(test_result[1], 4))
    fig.suptitle(data + num_data + network + param + metrics)
    fig.tight_layout()
    return plt.show()


# Dice score and loss function
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # tf.print(numerator, denominator)
    return tf.reduce_mean(numerator / (denominator+1))


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


# Load the dataset
# directory structure (Dropbox)
#   + ESUS_ML
#       + DCM_gtmaker_v2_release
#           + DCM: 603 input images (ADC & DWI); tiff files
#           + GT: 603 mask images (ADC & DWI); png files
#           + input: 74 input images (DWI); dcm files --> 0 masks with lesion
#       + DCM_gtmaker_v3
#           + DCM: 32,863 input images (ADC & DWI); tiff files --> local: 0
#           + GT: 32,863 mask images (ADC & DWI); png files --> local: 2,130
#           + input: 242 input images (ADC & DWI); dcm files --> 45 masks with lesion
#       + DCM_gtmaker_v5
#           + DCM: 7,216 input images (ADC & DWI); tiff files --> local: 7,135
#           + GT: 6,737 mask images (ADC & DWI); png files
#           + input: 574 input images (DWI); dcm files --> 525 non error images; 53 masks with lesion

# Get DWI image paths
fname_dwi = []  # 720 = 74+121+525
for dv in DCMv_dir:
    if dv == DCMv_dir[-1]:
        # Note: some of images are excluded due to the error below:
        # AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
        # e.g.
        # x_np[71] = './stroke_dcm/input\\281SVODWI0001.dcm'
        # x_np[224] = './stroke_dcm/input\\286SVODWI0001.dcm'
        fname_cut = get_matched_fpath(is_DWI_only, os.path.join(root_dir, dv))  # 574
        fname_dwi += fname_cut[:71] + fname_cut[97:224] + fname_cut[247:]  # 525
    else:
        fname_dwi += get_matched_fpath(is_DWI_only, os.path.join(root_dir, dv))

# Try a part of tiff files
if is_tiff:
    fname_dwi = fname_dwi[:round(len(fname_dwi)*0.2)]

# Separate the stroke subtypes
if is_subtype:
    fname_sub = separate_subtypes(fname_dwi)

    # Load the preprocessed image and mask datasets
    test_idx, x_all, y_all = load_images(is_DWI_only, fname_sub, rndcrop_size)

    # Shuffle the dataset and split into train and test sets
    '''
    x_train, y_train = shuffle_ds(x_all[:-test_idx[is_DWI_only][is_subtype][subtype_idx]],
                                  y_all[:-test_idx[is_DWI_only][is_subtype][subtype_idx]])
    x_valid, y_valid = x_all[-test_idx[is_DWI_only][is_subtype][subtype_idx]:], \
                       y_all[-test_idx[is_DWI_only][is_subtype][subtype_idx]:]
    '''
else:
    # Load the preprocessed image and mask datasets
    test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)  # 98 = 0+45+53

    # Shuffle the dataset and split into train and test sets
    # x_train, y_train = shuffle_ds(x_all[:-52], y_all[:-52])
    # x_valid, y_valid = x_all[-52:], y_all[-52:]  # (last 2 patients)

    # Shuffle the dataset and split into train and test sets
    # The entire dataset only includes the images with clearly delineated lesions on the mask
    '''
    x_train, y_train = shuffle_ds(x_all[:-test_idx[is_DWI_only][is_subtype]],
                                  y_all[:-test_idx[is_DWI_only][is_subtype]])
    x_valid, y_valid = x_all[-test_idx[is_DWI_only][is_subtype]:], \
                       y_all[-test_idx[is_DWI_only][is_subtype]:]
    '''

x_train, y_train = shuffle_ds(x_all[:-test_idx], y_all[:-test_idx])
x_valid, y_valid = x_all[-test_idx:], y_all[-test_idx:]

# Construct U-Net model
channel = 1 if is_DWI_only else 2
input_tensor = Input(shape=resize_size + (channel,), name='input_tensor')

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
test_result = u_net.evaluate(x_valid, y_valid, verbose=2)

# Generate the predicted result and plot it with the original image and mask
img = u_net.predict(x_valid)

'''
img_arg = np.argmax(img, axis=-1)
img_arg = img_arg[..., tf.newaxis]
img_arg = img_arg.astype('float32')
img_arg = img * 255
'''

# print(len(fname_dwi))
# print(len(x_all), len(y_all))

img_set = [x_valid, y_valid, img]
# ncol = test_idx[is_DWI_only][is_subtype] if not is_subtype else test_idx[is_DWI_only][is_subtype][subtype_idx]
ncol = test_idx
nrow = 3 if is_DWI_only else 4
show_img(is_DWI_only, img_set, ncol, nrow)

'''
fpath = './stroke_dcm/input/0102LAADWI0005.dcm'
ds = dcmread(fpath)

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
'''
