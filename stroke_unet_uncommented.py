#####
# Image semantic segmentation

# Dataset: stroke dicom files given by park
# 0 = background, 1 = lesion
# lesion is delineated on the mask file, which name contains 'DWI'

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import os
import re
import math
import matplotlib.pyplot as plt
from pydicom import dcmread, uid
from PIL import Image
from scipy.ndimage.interpolation import zoom
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate


# parameters
is_DWI_only = False  # DWI only if True else ADC+DWI
is_subtype = True  # one subtype only chosen by subtypes[subtype_idx] if True else all subtypes
subtypes = ['LAA', 'CE', 'SVO']
subtype_idx = 1
# is_tiff = True  # input data format == tiff if True else dicom

root_dir = 'D:/Dropbox/ESUS_ML/SSAI_STROKE'
# root_dir = 'F:\SSAI_STROKE'
# DCMv_dir = ['DCM_gtmaker_v2_release', 'DCM_gtmaker_v3', 'DCM_gtmaker_v5']

# for DWI only, last 5 images belong to one patient who are excluded in the train set (DCM_gtmaker_v5\GT\299SVODWI0013 ~ 17)
# for DWI only, last 3, 11, 5 images belong to laa, ce, svo patients respectively (laa: v5\GT\0120, ce: v3\GT\164, v5: v5\GT\299)
# for ADC+DWI, last 11 images (DCM_gtmaker_v3\GT\164CEDWI0011 ~ 21)
# for ADC+DWI, it contains ce patients only
# test_idx = 5 if is_DWI_only else 11
# test_idx = {True: {True: [3, 11, 5], False: 5}, False: {True: [0, 11, 0], False: 11}}  # test_idx[is_DWI_only][is_subtype]

max_dim = 256
depth = 4
pt_test_num = 4

rndcrop_size = (96, 96)
resize_size = rndcrop_size if isinstance(rndcrop_size, tuple) else (512, 512)
img_thld = 0.5
huber_weight = 10
learning_rate = 0.00005
batch_size = 3
epochs = 30
output_size = 1  # binary segmentation

# Results
# with about 10% of dataset
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

# (1) DWI_all *** : data = 98, learning rate = 0.0005, batch size = 3, epochs = 150 ==> test dice score = 0.67
# (1-1) DWI_LAA : data = 15, learning rate = 0.0005, batch size = 3, epochs = 150 ==> test dice score = 0.25
# (1-2) DWI_CE *** : data = 45, learning rate = 0.0005, batch size = 3, epochs = 150 ==> test dice score = 0.84
# (1-3) DWI_SVO : data = 38, learning rate = 0.0000005, batch size = 2, epochs = 150 ==> test dice score = 0.21
# (2) ADC+DWI_all (CE type only) *** : data = 45, learning rate = 0.0005, batch size = 3, epochs = 150 ==> test dice score = 0.85
# *** ==> a half of DWI only data is CE; DWI 절반 가량의 데이터가 CE환자; 다른 타입과 별개로 구분하여 학습할 때 더 좋은 결과
# *** ==> almost identical results with DWI and ADC+DWI

# with the entire dataset
# * 공통:
# - TRAIN DATA: mask에 병변부위 표시된 데이터만 입력, data size = 96*96
# - TEST DATA: all images belonging to the last pts not in train data; if num of images < 4, add one more pt's images
# - NETWORK: U-net, dimension = max. 256, depth = 4
# (1-1) ADC+DWI_LAA
# : data = 1360+4, learning rate = 0.0001, batch size = 2, epochs = 30 ==> test dice score = 0.62 (테스트 환자 2명)
# (1-2) ADC+DWI_CE
# : data = 1953+7, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.78 (테스트 환자 1명)
# (1-3) ADC+DWI_SVO
# : data = 518+4, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.77 (테스트 환자 2명)

# ==> with the entire dataset; increase the test set
# - TEST DATA: increase to 4 patients as test set and randomly plot 10 images among them
# (1) ADC + DWI_all
# : data = 3792+54, learning rate = 0.001, batch size = 16, epochs = 20 ==> test dice score = 0.81 (테스트 환자 12명)
# (1-1) ADC+DWI_LAA
# : data = 1343+21, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.87 (테스트 환자 4명)
# (1-2) ADC+DWI_CE
# : data = 1933+27, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.84 (테스트 환자 4명)
# (1-3) ADC+DWI_SVO
# : data = 516+6, learning rate = 0.000005, batch size = 2, epochs = 100 ==> test dice score = 0.46 (테스트 환자 4명)
# (2) DWI_all subtype
# : data = 3816+54, learning rate = 0.001, batch size = 16, epochs = 20 ==> test dice score = 0.81 (테스트 환자 12명)
# (2-1) DWI_LAA
# : data = 1346+21, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.88 (테스트 환자 4명)
# (2-2) DWI_CE
# : data = 1951+27, learning rate = 0.0001, batch size = 3, epochs = 30 ==> test dice score = 0.81 (테스트 환자 4명)

# Define necessary functions
def get_matched_fpath(is_DWI_only, mask_dir, img_dir):
    """ Return a list of path of input images containing 'DWI' in the 'input' folder under the given directory. """
    # Get file names in the folder
    (_, _, f_mask) = next(os.walk(mask_dir))
    (_, _, f_img) = next(os.walk(img_dir))

    f_mask.sort()
    f_img.sort()

    fname_dwi = []
    for i in f_img:
        if 'DWI' in i:
            f_dwi = os.path.splitext(i)[0]
            f_adc = f_dwi[:f_dwi.index('DWI')] + 'ADC' + f_dwi[f_dwi.index('DWI') + 3:]
            if is_DWI_only and (f_dwi + '.png' in f_mask):
                # fname_dwi has file names containing 'DWI', which have corresponding mask files
                fname_dwi.append(os.path.join(img_dir, i))
            elif (not is_DWI_only) and (f_adc + '.dcm' in f_img) and (f_dwi + '.png' in f_mask):
                # fname_dwi has file names containing 'DWI', which have corresponding 'ADC' and mask files
                fname_dwi.append(os.path.join(img_dir, i))

    return fname_dwi  # {folder_dir: fname_dwi}


'''
def separate_subtypes(fname_dwi):
    fname_sub = []
    for f_dwi in fname_dwi:
        f_base = os.path.basename(f_dwi)
        if subtypes[subtype_idx] in f_base:
            fname_sub.append(f_dwi)
    return fname_sub
'''


def load_images(is_DWI_only, fname_dwi, rndcrop_size):
    """
    Load the numpy array of preprocessed image and mask datasets
    :returns
    -
    """
    fname_pass, xs, ys = [], [], []
    for f in fname_dwi:
        try:
            # Load and rescale the mask images
            f_mask = os.path.join(os.path.dirname(os.path.dirname(f)), 'GT',
                                  os.path.splitext(os.path.basename(f))[0] + '.png')
            y = Image.open(f_mask).convert('L')  # from rgb to greyscale
            if isinstance(rndcrop_size, tuple):
                y = y.resize(rndcrop_size, resample=Image.BICUBIC)  # default resample = PIL.Image.BICUBIC
            # Make an index for the part of lesions as 1
            # (image thresholding) resize the image first and then apple thresholding
            y = y.point(lambda p: p > img_thld)  # *** 0.5 due to BICUBIC
            y = np.asarray(y, dtype='float32')

            # Collect only the masks where lesions are clearly delineated
            if y.max() == 0.:
                continue
            # Contains the file paths of DWI images that have lesion segmented masks
            # fname_pass.append(f)

            # Load and rescale the input images
            if is_DWI_only:
                '''
                if is_tiff:
                    x = Image.open(f)
                    x = x.resize(rndcrop_size, resample=Image.BICUBIC)
                    x = np.array(x, dtype='float32')
                    # *** normalization
                else:
                '''
                try:
                    x = dcmread(f).pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x = zoom(x, rndcrop_size[0] / x.shape[0])  # rescale
                    x = x.astype('float32') / 2048.0  # normalization
                except AttributeError:
                    # Resolve AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
                    x = dcmread(f)
                    x.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
                    x = x.pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x = zoom(x, rndcrop_size[0] / x.shape[0])  # rescale
                    x = x.astype('float32') / 2048.0  # normalization
            else:
                f_dwi = os.path.basename(f)
                f_adc = os.path.join(os.path.dirname(f),
                                     f_dwi[:f_dwi.index('DWI')] + 'ADC' + f_dwi[f_dwi.index('DWI') + 3:])
                '''
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
                '''
                try:
                    x_adc = dcmread(f_adc).pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x_adc = zoom(x_adc, rndcrop_size[0] / x_adc.shape[0])  # rescale
                    x_adc = x_adc.astype('float32') / 2048.0  # normalization
                except AttributeError:
                    # Resolve AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
                    x_adc = dcmread(f_adc)
                    x_adc.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
                    x_adc = x_adc.pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x_adc = zoom(x_adc, rndcrop_size[0] / x_adc.shape[0])  # rescale
                    x_adc = x_adc.astype('float32') / 2048.0  # normalization

                try:
                    x_dwi = dcmread(f).pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x_dwi = zoom(x_dwi, rndcrop_size[0] / x_dwi.shape[0])  # rescale
                    x_dwi = x_dwi.astype('float32') / 2048.0  # normalization
                except AttributeError:
                    # Resolve AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
                    x_dwi = dcmread(f)
                    x_dwi.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
                    x_dwi = x_dwi.pixel_array
                    if isinstance(rndcrop_size, tuple):
                        x_dwi = zoom(x_dwi, rndcrop_size[0] / x_dwi.shape[0])  # rescale
                    x_dwi = x_dwi.astype('float32') / 2048.0  # normalization

                # Concatenate ADC and DWI image
                x = np.concatenate((x_adc[:, :, np.newaxis], x_dwi[:, :, np.newaxis]), axis=-1)

            fname_pass.append(f)

        except Exception as e:
            print(f, e)

        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # Get the number of images belonging to last patient in fname_pass to put in test set
    pt_idx_last = re.findall('^\d+', os.path.basename(fname_pass[-1]))[0]
    pt_test = {pt_idx_last: 0}
    for i in range(len(fname_pass) - 1, 0, -1):
        pt_idx = re.findall('^\d+', os.path.basename(fname_pass[i]))[0]
        if pt_idx in pt_test.keys():
            pt_test[pt_idx] += 1
        elif len(pt_test.keys()) < pt_test_num:
            pt_test[pt_idx] = 1
        else:
            break
    test_idx = sum(pt_test.values())

    '''
    # TEST DATA: all images belonging to the last pts not in train data; if num of images < 4, add one more pt's images    
    test_idx = 0
    pt_idx_last = re.findall('^\d+', os.path.basename(fname_pass[-1]))[0]
    for i in range(len(fname_pass) - 2, 0, -1):
        pt_idx = re.findall('^\d+', os.path.basename(fname_pass[i]))[0]
        if (pt_idx != pt_idx_last) and (test_idx >= 4):
            break
        elif (pt_idx != pt_idx_last) and (test_idx < 4):
            pt_idx_last = re.findall('^\d+', os.path.basename(fname_pass[i]))[0]
        test_idx = len(fname_pass) - i
    '''
    return fname_pass, pt_test, test_idx, xs, ys


def shuffle_ds(x, y, z = None):
    """ Shuffle the train and test datasets (multi-dimensional array) along the first axis.
        Modify the order of samples in the datasets, while their contents and matching sequence remains the same. """
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    if type(z).__module__ == np.__name__:
        z = z[shuffle_idx]

    return x, y, z


def show_img(is_DWI_only, img_set, ncol, nrow):
    """ Plot the list of images consisting of input image, mask, and predicted result """
    fig = plt.figure(figsize=(8, 8))
    num_imgs = ncol * nrow
    ylabels = ['DWI ', 'Ground truth', 'AI prediction'] if is_DWI_only else ['ADC', 'DWI ', 'Ground truth', 'AI prediction']
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

    # data = ('TIFF ' if is_tiff else 'DICOM ') + \
    data = ('DICOM Data: DWI_' if is_DWI_only else 'Data: ADC+DWI_') + \
           (subtypes[subtype_idx] if is_subtype else 'all types')
    num_data = ' (' + str(len(x_train)) + ' in train, ' + str(len(x_valid)) + ' in test) \n'
    network = 'max dim: ' + str(max_dim) + ', depth: ' + str(depth) + '\n'
    param = 'Parameters: ' + str(rndcrop_size) + ', ' + str(learning_rate) + ', ' + str(batch_size) + ', ' + str(epochs) + '\n'
    # metrics = 'Test dice loss & score: ' + str(round(test_result[0], 4)) + ', ' + str(round(test_result[1], 4))
    fig.suptitle(data + num_data + network + param)  #+ metrics)
    fig.tight_layout()

    fig.set_size_inches(15, 7)
    plt.savefig('stroke_w_lesion__' +
                ('DWI-' if is_DWI_only else 'ADCDWI-') +
                (subtypes[subtype_idx] if is_subtype else 'all') +
                ('-' + str(len(x_train)) + '+' + str(len(x_valid))) +
                ('__net-' + str(max_dim) + '.' + str(depth)) +
                ('__' + ((str(rndcrop_size[0]) + '.' + str(rndcrop_size[1]) if isinstance(rndcrop_size, tuple) else str(rndcrop_size))) +
                 '_' + str(learning_rate) + '_' + str(batch_size) + '_' +  str(epochs)) + '.png')  # +
                # ('__dice-' + str(round(test_result[1], 4))) + '.png')

    return plt.show()


# Dice score and loss function
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # tf.print(numerator, denominator)
    return tf.reduce_mean(numerator / (denominator + 1))


def dice_loss(y_true, y_pred):
    huber = tf.losses.Huber()
    huber = huber(y_true, y_pred)
    return 1 - dice_score(y_true, y_pred) + huber_weight*huber


# Load the dataset
# directory structure (Dropbox)
#   + ESUS_ML
#       + SSAI_STROKE
#           + LAA: 30,567 input images (ADC & DWI); dcm files; originally from DATA set + DATAset_20200324
#           + CE: 21,905 input images (ADC & DWI); dcm files; originally from DATA set + DATAset_20200324
#           + SVO: 18,024 input images (ADC & DWI); dcm files; originally from DATA set + DATAset_20200324
#           + GT: 38,328 mask images (ADC & DWI); png files; originally from DCM_gtmaker_v2,3,5
#           + Control: 17,610 images; originally from DATA set

# Get DWI image paths
# fname_dwi = []  # 720 = 74+121+525
mask_dir = os.path.join(root_dir, 'GT')
if is_subtype:
    img_dir = os.path.join(root_dir, subtypes[subtype_idx])
    fname_dwi = get_matched_fpath(is_DWI_only, mask_dir, img_dir)
    fname_pass, pt_test, test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)

    # fname_pass, pt_test, test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)
    x_train, y_train, _ = shuffle_ds(x_all[:-test_idx], y_all[:-test_idx])
    x_valid, y_valid = x_all[-test_idx:], y_all[-test_idx:]

else:
    test_idx = []
    for sub in subtypes:
        img_dir = os.path.join(root_dir, sub)
        fname_dwi = get_matched_fpath(is_DWI_only, mask_dir, img_dir)
        if sub == 'LAA':
            fname_pass, pt_test, t_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)
            test_idx.append(t_idx)

            # fname_pass, pt_test, test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)
            x_train, y_train = x_all[:-t_idx], y_all[:-t_idx]
            x_valid, y_valid = x_all[-t_idx:], y_all[-t_idx:]
        else:
            f_pass, pt_test, t_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)
            fname_pass += f_pass
            # pt_test.update(b)
            test_idx.append(t_idx)

            # fname_pass, pt_test, test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)
            x_train = np.concatenate((x_train, x_all[:-t_idx]), axis=0)
            y_train = np.concatenate((y_train, y_all[:-t_idx]), axis=0)
            x_valid = np.concatenate((x_valid, x_all[-t_idx:]), axis=0)
            y_valid = np.concatenate((y_valid, y_all[-t_idx:]), axis=0)

    x_train, y_train, _ = shuffle_ds(x_train, y_train)

'''
for dv in DCMv_dir:
    """
    # Fixed the issue below 
    if dv == DCMv_dir[-1]:
        # Note: some of images are excluded due to the error below:
        # AttributeError: 'FileMetaDataset' object has no attribute 'TransferSyntaxUID'
        # e.g.
        # x_np[71] = './stroke_dcm/input\\281SVODWI0001.dcm'
        # x_np[224] = './stroke_dcm/input\\286SVODWI0001.dcm'
        fname_cut = get_matched_fpath(is_DWI_only, os.path.join(root_dir, dv))  # 574
        fname_dwi += fname_cut[:71] + fname_cut[97:224] + fname_cut[247:]  # 525
    else:
    """
    fname_dwi += get_matched_fpath(is_DWI_only, os.path.join(root_dir, dv))
'''

'''
# Try a part of tiff files
if is_tiff:
    fname_dwi = fname_dwi[:round(len(fname_dwi)*0.2)]
'''
"""
# Separate the stroke subtypes
if is_subtype:
    # fname_sub = separate_subtypes(fname_dwi)

    # Load the preprocessed image and mask datasets
    fname_pass, test_idx, x_all, y_all = load_images(is_DWI_only, fname_sub, rndcrop_size)

    # Shuffle the dataset and split into train and test sets
    '''
    x_train, y_train = shuffle_ds(x_all[:-test_idx[is_DWI_only][is_subtype][subtype_idx]],
                                  y_all[:-test_idx[is_DWI_only][is_subtype][subtype_idx]])
    x_valid, y_valid = x_all[-test_idx[is_DWI_only][is_subtype][subtype_idx]:], \
                       y_all[-test_idx[is_DWI_only][is_subtype][subtype_idx]:]
    '''
else:
    # Load the preprocessed image and mask datasets
    fname_pass, test_idx, x_all, y_all = load_images(is_DWI_only, fname_dwi, rndcrop_size)  # 98 = 0+45+53

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
"""

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
u_net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

# Test the model with test set
test_result = u_net.evaluate(x_valid, y_valid, verbose=1)

# Generate the predicted result and plot it with the original image and mask
img = u_net.predict(x_valid)

# Plot the test result
#img_idx = [3,4,11,12,17,18,19,20]  # LAA
#img_idx = [4,5,10,14,15,22,23]  # CE
#img_set = [x_valid[img_idx], y_valid[img_idx], img[img_idx]]
a, b, c = shuffle_ds(x_valid, y_valid, img)
#img_set = [a[:math.floor(len(a)/2)], b[:math.floor(len(a)/2)], c[:math.floor(len(a)/2)]]
img_set = [a[:10], b[:10], c[:10]]
# ncol = test_idx[is_DWI_only][is_subtype] if not is_subtype else test_idx[is_DWI_only][is_subtype][subtype_idx]
#ncol = len(img_idx)
#ncol = math.floor(len(a)/2)
ncol = len(a) if len(a) < 10 else 10
nrow = 3 if is_DWI_only else 4
show_img(is_DWI_only, img_set, ncol, nrow)


#####
plt.pcolor(img[0][:,:,0])
#####
from matplotlib.colors import Normalize, LinearSegmentedColormap

# transparency
alphas = Normalize(0, .3, clip=True)(np.abs(img[0][:,:,0]))
alphas = np.clip(alphas, .4, 1)

fig, ax = plt.subplots()
ax.imshow(x_valid[0][:,:,0])
ax.imshow(img[0], alpha=alphas)

# colormap + transparency
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))
color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)


fig, ax = plt.subplots()
ax.imshow(x_valid[0][:,:,0], cmap=plt.cm.gray)
h = ax.imshow(img[0], cmap='rainbow_alpha')
plt.colorbar(mappable=h)

# outline
ax.contour(img[0][:,:,0], levels=[-.1, .1], colors='k', linestyles='-')  # darker background
ax.set_axis_off()


'''
img_arg = np.argmax(img, axis=-1)
img_arg = img_arg[..., tf.newaxis]
img_arg = img_arg.astype('float32')
img_arg = img * 255
'''
'''
fpath = './stroke_dcm/input/0102LAADWI0005.dcm'
ds = dcmread(fpath)

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
'''
