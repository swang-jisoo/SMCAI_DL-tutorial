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

# Hyper-parameters

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

data_dir = './dataverse_files'

def get_tar_fname(data_dir, tar):
    img_tar, img_fname = [], []
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            img_fname.append(i_tar.getnames())
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


xs, ys = [], []

for i in range(len(img_tar)):
    for j in img_fname[i][1:]:
        img = img_tar[i].extractfile(j)
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
        y = y.crop(roi)

        x = np.asarray(x, dtype='float32') / 255.0
        y = np.asarray(y, dtype='float32') / 255.0

        # crop -> resize, increase sample size

        xs.append(x)
        ys.append(y)

x_np = np.asarray(xs)
y_np = np.asarray(ys)
