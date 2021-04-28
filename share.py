
# Import necessary libraries
# Import necessary libraries
import os
import re
from pydicom import dcmread, uid
from PIL import Image
from scipy.ndimage.interpolation import zoom
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Conv3D, UpSampling3D, BatchNormalization, Dense, Activation, \
    MaxPool3D, AveragePooling3D, GlobalMaxPool3D, GlobalAveragePooling3D


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


def U_NET(data_shape, output_size=1, kernel_size=3, dim=32):
    input = Input(data_shape, name='input')

    # Encoder
    # block 1
    conv1_1 = Conv2D(dim, kernel_size, activation='relu', padding='same', name='conv1_1')(input)
    conv1_2 = Conv2D(dim, kernel_size, activation='relu', padding='same', name='conv1_2')(conv1_1)
    max1 = MaxPooling2D(2, padding='same', name='max1')(conv1_2)

    # block 2
    conv2_1 = Conv2D(2 * dim, kernel_size, activation='relu', padding='same', name='conv2_1')(max1)
    conv2_2 = Conv2D(2 * dim, kernel_size, activation='relu', padding='same', name='conv2_2')(conv2_1)
    max2 = MaxPooling2D(2, padding='same', name='max2')(conv2_2)

    # block 3
    conv3_1 = Conv2D(4 * dim, kernel_size, activation='relu', padding='same', name='conv3_1')(max2)
    conv3_2 = Conv2D(4 * dim, kernel_size, activation='relu', padding='same', name='conv3_2')(conv3_1)
    max3 = MaxPooling2D(2, padding='same', name='max3')(conv3_2)

    # block 4
    conv4_1 = Conv2D(8 * dim, kernel_size, activation='relu', padding='same', name='conv4_1')(max3)
    conv4_2 = Conv2D(8 * dim, kernel_size, activation='relu', padding='same', name='conv4_2')(conv4_1)
    # max4 = MaxPooling2D(2, padding='same', name='max4')(conv4_2)
    #
    # # block 5
    # conv5_1 = Conv2D(16 * dim, kernel_size, activation='relu', padding='same', name='conv5_1')(max4)
    # conv5_2 = Conv2D(16 * dim, kernel_size, activation='relu', padding='same', name='conv5_2')(conv5_1)
    #
    # # Decoder
    # # block 4
    # deconv4_1 = Conv2DTranspose(8 * dim, 2, 2, activation='relu', name='deconv4_1')(conv5_2)
    # merge4 = tf.concat([deconv4_1, conv4_2], axis=-1, name='merge4')
    # deconv4_2 = Conv2D(8 * dim, kernel_size, activation='relu', padding='same', name='deconv4_2')(merge4)
    # deconv4_3 = Conv2D(8 * dim, kernel_size, activation='relu', padding='same', name='deconv4_3')(deconv4_2)

    # block 3
    deconv3_1 = Conv2DTranspose(4 * dim, 2, 2, activation='relu', name='deconv3_1')(conv4_1)
    merge3 = tf.concat([deconv3_1, conv3_2], axis=-1, name='merge3')
    deconv3_2 = Conv2D(4 * dim, kernel_size, activation='relu', padding='same', name='deconv3_2')(merge3)
    deconv3_3 = Conv2D(4 * dim, kernel_size, activation='relu', padding='same', name='deconv3_3')(deconv3_2)

    # block 2
    deconv2_1 = Conv2DTranspose(2 * DIM, 2, 2, activation='relu', name='deconv2_1')(deconv3_3)
    merge2 = tf.concat([deconv2_1, conv2_2], axis=-1, name='merge2')
    deconv2_2 = Conv2D(2 * dim, kernel_size, activation='relu', padding='same', name='deconv2_2')(merge2)
    deconv2_3 = Conv2D(2 * dim, kernel_size, activation='relu', padding='same', name='deconv2_3')(deconv2_2)

    # block 1
    deconv1_1 = Conv2DTranspose(dim, 2, 2, activation='relu', name='deconv1_1')(deconv2_3)
    merge1 = tf.concat([deconv1_1, conv1_2], axis=-1, name='merge1')
    deconv1_2 = Conv2D(dim, kernel_size, activation='relu', padding='same', name='deconv1_2')(merge1)
    deconv1_3 = Conv2D(dim, kernel_size, activation='relu', padding='same', name='deconv1_3')(deconv1_2)

    output = Conv2D(output_size, 1, activation='sigmoid', padding='same', name='output')(deconv1_3)

    model = Model(input, output, name='u_net')

    return model


def ResBlock(src, kernel_size, dim, isattention):

    conv = Conv3D(dim, kernel_size, activation=None, padding='same')(src)

    bn1 = BatchNormalization()(conv)
    relu1 = tf.nn.relu(bn1)
    conv1 = Conv3D(dim, kernel_size, activation=None, padding='same')(relu1)

    bn2 = BatchNormalization()(conv1)
    relu2 = tf.nn.relu(bn2)
    conv2 = Conv3D(dim, kernel_size, activation=None, padding='same')(relu2)

    return conv + conv2


def BN_relu_Conv(src, kernel_size, dim):

    bn = BatchNormalization()(src)
    relu = tf.nn.relu(bn)
    conv = Conv3D(dim, kernel_size, activation=None, padding='same')(relu)

    return conv

def Res_V_NET(data_shape, kernel_size=3, dim=8, isattention='None'):

    src = Input(data_shape)

    # block 1
    conv1 = ResBlock(src, kernel_size, dim, isattention)
    max1 = MaxPool3D((1, 2, 2), padding='same')(conv1)

    # block 2
    conv2 = ResBlock(max1, kernel_size, 2 * dim, isattention)
    max2 = MaxPool3D(2, padding='same')(conv2)

    # block 3
    conv3 = ResBlock(max2, kernel_size, 4*dim, isattention)
    max3 = MaxPool3D((1, 2, 2), padding='same')(conv3)

    # block 4
    conv4 = ResBlock(max3, kernel_size, 8*dim, isattention)
    max4 = MaxPool3D(2, padding='same')(conv4)

    # block 5
    conv5 = ResBlock(max4, kernel_size, 16*dim, isattention)

    # block 4
    up4 = UpSampling3D(2)(conv5)
    deconv4_1 = BN_relu_Conv(up4, kernel_size, 8*dim)
    merge4 = tf.concat([deconv4_1, conv4], axis=-1)
    deconv4_2 = ResBlock(merge4, kernel_size, 8*dim, isattention)

    # block 3
    up3 = UpSampling3D((1, 2, 2))(deconv4_2)
    deconv3_1 = BN_relu_Conv(up3, kernel_size, 4*dim)
    merge3 = tf.concat([deconv3_1, conv3], axis=-1)
    deconv3_2 = ResBlock(merge3, kernel_size, 4*dim, isattention)

    # block 2
    up2 = UpSampling3D(2)(deconv3_2)
    deconv2_1 = BN_relu_Conv(up2, kernel_size, 2 * dim)
    merge2 = tf.concat([deconv2_1, conv2], axis=-1)
    deconv2_2 = ResBlock(merge2, kernel_size, 2 * dim, isattention)

    # block 1
    up1 = UpSampling3D((1, 2, 2))(deconv2_2)
    deconv1_1 = BN_relu_Conv(up1, kernel_size, dim)
    merge1 = tf.concat([deconv1_1, conv1], axis=-1)
    deconv1_2 = ResBlock(merge1, kernel_size, dim, isattention)

    pred = Conv3D(1, kernel_size, activation='sigmoid', padding='same')(deconv1_2)

    model = Model(src, pred)

    return model

