# Import necessary libraries
import os
import tarfile
from PIL import Image
import numpy as np
import math
import random

# Hyper-parameters
rnd_freq = 10  # number of randomly cropped images to generate
rndcrop_size = (216, 216)  # W, H
resize_size = (3100, 3100)   # W, H

data_dir = 'D:/PycharmProjects/dataverse_files' # 변경 필요
train_tar = ['Gleason_masks_train', 'ZT76_39_A', 'ZT76_39_B', 'ZT111_4_A', 'ZT111_4_B', 'ZT111_4_C',
             'ZT199_1_A', 'ZT199_1_B', 'ZT204_6_A', 'ZT204_6_B']
test_tar1 = ['Gleason_masks_test_pathologist1',  # 'Gleason_masks_test_pathologist2',
            'ZT80_38_A', 'ZT80_38_B', 'ZT80_38_C']
test_tar2 = ['Gleason_masks_test_pathologist2',
            'ZT80_38_A', 'ZT80_38_B', 'ZT80_38_C']

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

mask_tar1, img_tar, mask_fname1, img_fname = get_tar_fname(data_dir, test_tar1)
mask_tar2, _, mask_fname2, _ = get_tar_fname(data_dir, test_tar2)

img_fname.sort()
mask_fname1.sort()
mask_fname2.sort()

xs, y1s, y2s = [], [], []
for m in range(len(img_fname)):
    tar_idx = test_tar1.index(os.path.dirname(img_fname[m])) - 1
    x = img_tar[tar_idx].extractfile(img_fname[m])
    y1 = mask_tar1.extractfile(mask_fname1[m])
    y2 = mask_tar2.extractfile(mask_fname2[m])

    x = Image.open(x)
    y1 = Image.open(y1)
    y2 = Image.open(y2)

    xs.append(x)
    y1s.append(y1)
    y2s.append(y2)

'''x_trial = np.asarray(x)
y1_trial = np.asarray(y1) * 50
y2_trial = np.asarray(y2) * 50

# input original image
img_x = Image.fromarray(x_trial, 'RGB')
img_x.show()

# mask image in grey scale
img_y1 = Image.fromarray(y1_trial, 'L')
img_y1.show()
img_y2 = Image.fromarray(y2_trial, 'L')
img_y2.show()'''

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

xy_show = []
xy_show.append(xs)
xy_show.append(y1s)
xy_show.append(y2s)

imgs = show_img(xy_show, 5, 6, resize_size)
imgs.show()
