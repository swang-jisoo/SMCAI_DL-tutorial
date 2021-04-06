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
import numpy as np
import pydicom
import cv2
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from datetime import date

# parameters
# NTEST = 30
DATA_SHAPE = (256, 256, 2)
PER_VAL = 0.2
OUTPUT_SIZE = 1
DIM = 16
KERNEL_SIZE = 3
HUBER_WEIGHT = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 3
EPOCHS = 60
VERBOSE = 1

today = ''.join(str(date.today()).split('-'))[2:]
checkpoint_path = "C:/Users/swang/PycharmProjects/SSAI/model/stroke_unet_%s.ckpt" % today
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=10)

# Results
# DATA_SHAPE = (128, 128, 2), OUTPUT_SIZE = 1, DIM = 16, KERNEL_SIZE = 3, LEARNING_RATE = 1e-4, BATCH_SIZE = 2, EPOCHS = 30; loss: 0.1303 - dice_score: 0.8727


# Define necessary functions
# data functions
def load_pickle(pickle_path):

    f = open(pickle_path, 'rb')
    data = pickle.load(f)

    return data


def save_pickle(pickle_path, arg):

    f = open(pickle_path, 'wb')
    pickle.dump(arg, f)
    f.close()

    print('SAVING PICKLE FINISH')


def data_split(src, gt, subtype, test_p = 0.2):

    num = len(src)
    n_test = (np.round(num * test_p)).astype(int)
    n_train = num - n_test

    src_test = src[:n_test, :, :, :, :]
    src_train = src[n_test:, :, :, :, :]
    gt_test = gt[:n_test, :, :, :, :]
    gt_train = gt[n_test:, :, :, :, :]

    label = {"LAA": 0.0, "CE": 1.0, "SVO": 2.0, "CTRL": 3.0}
    lb_train = label[subtype] * np.ones((n_train, 1), dtype=float)
    lb_test = label[subtype] * np.ones((n_test, 1), dtype=float)

    print("num_train: %d, num_test: %d" % (n_train, n_test))

    return [src_train, gt_train, lb_train, src_test, gt_test, lb_test]


def convert_2D(x, y):
    data, gt, pt_slide = [], [], {}
    for i in range(len(x)):
        for j in range(x.shape[1]):
            A = np.sum(y[i, j, :, :, 0])
            if(A >= 25):
                if len(data) == 0:
                    data = np.expand_dims(x[i, j, :, :, :], axis=0)
                    gt = np.expand_dims(y[i, j, :, :, 0], axis=0)
                else:
                    data = np.append(data, np.expand_dims(x[i, j, :, :, :], axis=0), axis=0)
                    gt = np.append(gt, np.expand_dims(y[i, j, :, :, 0], axis=0), axis=0)
                if i in pt_slide:
                    pt_slide[i] += 1
                else:
                    pt_slide[i] = 1
    gt = gt[:, :, :, np.newaxis]
    return data, gt, pt_slide


# model functions
def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def attention(self, x, ch):
    f = Conv2D(ch // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')(x)  # [bs, h, w, c']
    g = Conv2D(ch // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')(x)  # [bs, h, w, c']
    h = Conv2D(ch, kernel=1, stride=1, sn=self.sn, scope='h_conv')(x)  # [bs, h, w, c]

    # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
    x = gamma * o + x

    return x


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


# loss functions / metrics
def weighted_binary_crossentropy(y_true, y_pred):

    one_weight = 0.95
    zero_weight = 0.05
    b_ce = K.binary_crossentropy(y_true, y_pred)

    # weighted calc
    weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    return K.mean(weighted_b_ce)


def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # tf.print(numerator, denominator)
    return tf.reduce_mean(numerator / (denominator + 1))


def dice_loss(y_true, y_pred):
    # huber = tf.losses.Huber()
    # huber = huber(y_true, y_pred)
    return 1 - dice_score(y_true, y_pred)  # + HUBER_WEIGHT * huber


# class MyCustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, batch, logs=None):
#         if ncount[0] % 10 == 0:
#             os.makedirs("./%04d" % ncount[0])
#             img = np.zeros((DATA_SHAPE[1], 3 * DATA_SHAPE[2]), dtype='uint8')
#             for i in range(len(test_data)):
#                 tsrc = np.expand_dims(test_data[i, :, :, ::], 0)
#                 tgt = np.expand_dims(test_gt[i, :, :, :, :], 0)
#                 y_pred = model.predict(tsrc)
#                 for j in range(DATA_SHAPE[0]):
#                     img[:, :DATA_SHAPE[2]] = (255 * tgt[0, j, :, :, 0]).astype('uint8')
#                     img[:, DATA_SHAPE[2]:2 * DATA_SHAPE[2]] = (255 * y_pred[0, j, :, :, 0]).astype('uint8')
#                     img[:, 2 * DATA_SHAPE[2]:] = (255 * tsrc[0, j, :, :, 0]).astype('uint8')
#                     cv2.imwrite("./%04d/%02d_%02d.png" % (ncount[0], i, j), img)
#         ncount[0] = ncount[0] + 1


# Load the dataset
# 256 x 256, bicubic intepolation
# [LAA_data, LAA_gt, CE_data, CE_gt, SVO_data, SVO_gt, CTRL_data, CTRL_gt] = load_pickle('C:\\Users\\swang\\PycharmProjects\\data - stroke\\pickle_wholedata.pkl')

# [src11, gt11, lb11, src12, gt12, lb12] = data_split(LAA_data, LAA_gt, "LAA", test_p=PER_VAL)  # num_train: 486, num_test: 122
# [src21, gt21, lb21, src22, gt22, lb22] = data_split(SVO_data, SVO_gt, "SVO", test_p=PER_VAL)  # num_train: 353, num_test: 88
# [src31, gt31, lb31, src32, gt32, lb32] = data_split(CE_data, CE_gt, "CE", test_p=PER_VAL)  # num_train: 287, num_test: 72
# [src41, gt41, lb41, src42, gt42, lb42] = data_split(CTRL_data, CTRL_gt, "CTRL", test_p=PER_VAL)  # num_train: 320, num_test: 80

# train_data = np.concatenate([src11, src21, src31], axis=0)  # 1126
# train_gt = np.concatenate([gt11, gt21, gt31], axis=0)
# train_lb = np.concatenate([lb11, lb21, lb31], axis=0)
# test_data = np.concatenate([src12, src22, src32], axis=0)  # 282
# test_gt = np.concatenate([gt12, gt22, gt32], axis=0)
# test_lb = np.concatenate([lb12, lb22, lb32], axis=0)

# x_train, y_train, pt_slide_train = convert_2D(train_data, train_gt)  # 5261
# x_test, y_test, pt_slide_test = convert_2D(test_data, test_gt)  # 1258
#
# train_lb, test_lb = np.array(train_lb), np.array(test_lb)
# lb_train = train_lb[list(pt_slide_train)]
# lb_test = test_lb[list(pt_slide_test)]
#
# save_pickle('C:/Users/swang/PycharmProjects/data - stroke/pickle_segdata_2d.pkl', [x_train, y_train, lb_train, x_test, y_test, lb_test])
[x_train, y_train, lb_train, x_test, y_test, lb_test] = load_pickle('C:/Users/swang/PycharmProjects/data - stroke/pickle_segdata_2d.pkl')
y_train, y_test = y_train.astype('float32'), y_test.astype('float32')  # bool --> float32

# Construct U-Net model
model = U_NET(data_shape=DATA_SHAPE, output_size=OUTPUT_SIZE)

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss=weighted_binary_crossentropy, metrics=[dice_score])
model.summary()

# Train the model to adjust parameters to minimize the loss
model.load_weights(checkpoint_path)
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[cp_callback])  # validation_split=PER_VAL,

# Generate the predicted result and plot it with the original image and mask
# model.load_weights(checkpoint_path)
# epochs = 20+20(210406:17:11, 0.75)
pred = model.predict(x_test)


# Plot the test result
show_idx = [i for i in range(10)]  # [1,15,21,35,45,57,68.73,85,91]

# ytitle = ["DWI", "Ground Truth", "AI Prediction"]
ytitle = ["ADC", "DWI", "Ground Truth", "AI Prediction"]
nrow, ncol = len(ytitle), len(show_idx)

# file name to save
today = ''.join(str(date.today()).split('-'))[2:]
save_name = 'stroke_3DCNN_results_%s.png' % (today)

fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow))

for i in show_idx:
    gt = y_test[i, :, :, 0]
    gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    # gt = (gt * 256).astype('uint8')
    seg = pred[i, :, :, 0]
    seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)
    seg = (seg * 256).astype('uint8')
    dwi = x_test[i, :, :, 0]
    dwi = cv2.cvtColor(dwi, cv2.COLOR_GRAY2RGB)
    dwi = (dwi * 256).astype('uint8')
    adc = x_test[i, :, :, 1]
    adc = cv2.cvtColor(adc, cv2.COLOR_GRAY2RGB)
    adc = (adc * 256).astype('uint8')

    # GT
    gt_mask = [0, 0, 0] * np.ones(shape=dwi.shape, dtype='uint8')
    # gt_mask = np.copy(gt)
    gt_mask[(gt > 0).all(-1)] = [128, 0, 0]
    gt_mask = gt_mask.astype('uint8')
    gt_result = cv2.addWeighted(dwi, 1, gt_mask, 0.4, 0)

    # Prediction
    seg_mask = cv2.cvtColor(cv2.applyColorMap(seg, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    seg_mask[(seg_mask == [0, 0, 128]).all(-1)] = [0, 0, 0]
    seg_dwi_result = cv2.addWeighted(dwi, 1, seg_mask, 0.4, 0)

    # ximg = [dwi, gt_result, seg_dwi_result]
    ximg = [adc, dwi, gt_result, seg_dwi_result]
    for j in range(nrow):
        ax[j, i].imshow(ximg[j])
        if i == 0:
            ax[j, i].tick_params(axis='both', which='both', length=0)
            ax[j, i].axes.xaxis.set_ticklabels([])
            ax[j, i].axes.yaxis.set_ticklabels([])
            ax[j, i].set_ylabel(ytitle[j])
        else:
            ax[j, i].axis('off')

plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=1.0)

today = ''.join(str(date.today()).split('-'))[2:]
plt.savefig('stroke_2DUNET_results_%s.png' % (today))

