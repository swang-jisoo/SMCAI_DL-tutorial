#####
# Image semantic segmentation

# Dataset: harvard dataverse prostate
# ref. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Import necessary libraries
import os
import tarfile
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate

# Hyper-parameters
learning_rate = 0.05  # for u-net, start with larger learning rate
batch_size = 16
epochs = 20

# Results
# ==>

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

def get_tar_fname(data_dir, tar):
    img_tar, img_fname = [], []
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()[1:]
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            img_fname.append(i_tar.getnames()[1:])
    return mask_tar, img_tar, mask_fname, img_fname

mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, train_tar)

mask_fname_match, img_fname_match = [], []

for m in range(len(mask_fname)):
    m_fname = os.path.splitext(os.path.basename(mask_fname[m]))[0]
    m_fname_split = m_fname.split('_')
    if 'mask' not in m_fname_split[0]:
        continue
    tar_idx = train_tar.index('_'.join(m_fname_split[1:4]))
    for i in range(len(img_fname[tar_idx - 1])):
        i_fname = os.path.splitext(os.path.basename(img_fname[tar_idx - 1][i]))[0]
        if '_'.join(m_fname_split[1:]) == i_fname:
            mask_fname_match.append(mask_fname[m])
            img_fname_match.append(img_fname[tar_idx - 1][i])

'''
# 수정 중

def get_tar_fname(data_dir, tar):
    img_tar, img_fname = [], {}
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            i_fname = i_tar.getnames()
            img_fname[i_fname[0]] = i_fname[1:]
    return mask_tar, img_tar, mask_fname, img_fname

mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, train_tar)

tar = train_tar
mask_fname_match, img_fname_match = [], {}
for t in tar[1:]:
    i_fname_match = []
    for i in img_fname[t]:
        i_matcher = os.path.splitext(os.path.basename(i))[0]
        for m in mask_fname[1:]:
            m_matcher = os.path.splitext(os.path.basename(m))[0]
            if 'mask' not in m_fname_split[:4]:
                continue
            elif i_matcher == m_matcher[-len(i_matcher)+1:]:
                mask_fname_match.append(m)
                i_fname_match.append(i)
    img_fname_match[t] = i_fname_match
'''

'''
# 안 쓸 예정
tar = train_tar
mask_fname_match = []
xs, ys = [], []
for t in range(len(tar)):
    tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
    if t == 0:
        mask_tar = tarfile.open(tar_fname)
        mask_fname = tar_open.getnames()
    else:
        img_tar = tarfile.open(tar_fname)
        img_fname = tar_open.getnames()

    for i in img_fname:
        matcher = os.path.splitext(os.path.basename(i))[0]  # i[len(tar[t])+1:-4]
        for m in mask_fname:
            if '/mask_' + matcher + '.png' in m:
                mask_fname_match.append(m)
                img_fname_match.append(i)

    print(len(mask_fname_match), len(img_fname_match))

    for i in img_fname_match:
        img = img_tar.extractfile(i)

        x = Image.open(img)
        x = np.array(x)
        xs.append(x)
'''

# mask image open here
'''
# 노쓸모
mask_tar_fname = './dataverse_files/' + train_tar[0] + '.tar.gz'
mask_tar = tarfile.open(mask_tar_fname)
mask_fname = mask_tar.getnames()

img_tar_fname = './dataverse_files/' + train_tar[1] + '.tar.gz'
img_tar = tarfile.open(img_tar_fname)
img_fname = img_tar.getnames()

mask_match, img_match = [], []
for n in img_fname:
    matcher = n[len(train_tar[1])+1:-4]
    for m in mask_fname:
        if '/mask_' + matcher + '.png' in m:
            mask_match.append(m)
            img_match.append(n)
'''

'''name_match = []
matcher = img_fname[1][len(train_tar[1])+1:-4]
if matcher in mask_fname:
    name_match.append(matcher)'''

ori_size = (216, 216)
img_size = (72, 72)
img_fname_match.sort()

def load_images(is_train, length, ori_size, img_size):
    xs = []

    for i in range(1): # range(len(img_tar)):
        for j in range(length):  # img_fname[i]:
            if is_train:
                img = img_tar[2].extractfile(img_fname_match[j]) # img_tar랑 img_fname_folder 명이랑 매치 -> dict
            else:
                img = mask_tar.extractfile(mask_fname_match[j])  # mask_tar and mask_fname
            x = Image.open(img)
            '''    
    for f in img_fname:
        img = img_tar.extractfile(f)
    
        x = Image.open(img)
            '''
            # Preprocessing (crop and normalize)
            width, height = x.size
            cx = int(width / 2)
            cy = int(height / 2)

            roi = (cx - ori_size[0] / 2, cy - ori_size[1] / 2, cx + ori_size[0] / 2, cy + ori_size[1] / 2)

            x = x.crop(roi)
            #y = y.crop(roi)

            x = x.resize(img_size)

            x = np.asarray(x, dtype='float32') / 255.0
            #y = np.asarray(y, dtype='float32') / 255.0

            # crop -> resize, increase sample size

            xs.append(x)
            #ys.append(y)
    return xs

xs = load_images(True, 70, ori_size, img_size)
ys = load_images(False, 70, ori_size, img_size)
x_train = np.asarray(xs)
y_train = np.asarray(ys)




# Construct U-Net model
# *** input shape
input_tensor = Input(shape=img_size + (3,), name='input_tensor')

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
output_tensor = Conv2D(20 + 1, 1, padding='same', activation='sigmoid', name='output_tensor')(expn4_2)

# Create a model
u_net = Model(input_tensor, output_tensor, name='u_net')
u_net.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train, epochs=epochs)

# Test the model with test set
#u_net.evaluate(x_valid, y_valid, verbose=2)
