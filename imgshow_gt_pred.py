# png -> cut and overlay
import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from datetime import date

# path = "C:\\Users\\swang\\PycharmProjects\\results - stroke\\3D CNN storke_result_20210405\\24_16.png"
paths = glob.glob("C:\\Users\\swang\\PycharmProjects\\results - stroke\\3D CNN storke_result_20210405\\*.png")
save_path = os.path.dirname(paths[0])
nrow, ncol = 2, len(paths)
fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow))

for i in range(len(paths)):
    img = cv2.imread(paths[i], cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    w_cut = w // 3
    gt, seg, dwi = img[:, :w_cut, :], img[:, w_cut:w_cut*2, :], img[:, w_cut*2:, :]

    # GT
    gt_mask = np.copy(gt)
    gt_mask[(gt > 0).all(-1)] = [128, 0, 0]
    gt_result = cv2.addWeighted(dwi, 1, gt_mask, 0.3, 0)

    # Prediction
    seg_mask = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
    seg_result = cv2.cvtColor(cv2.addWeighted(dwi, 1, seg_mask, 0.3, 0), cv2.COLOR_BGR2RGB)

    ax[0, i].imshow(gt_result)
    ax[1, i].imshow(seg_result)
    if i == 0:
        ax[0, i].tick_params(axis='both', which='both', length=0)
        ax[0, i].axes.xaxis.set_ticklabels([])
        ax[0, i].axes.yaxis.set_ticklabels([])
        ax[1, i].tick_params(axis='both', which='both', length=0)
        ax[1, i].axes.xaxis.set_ticklabels([])
        ax[1, i].axes.yaxis.set_ticklabels([])
        ax[0, i].set_ylabel("GT")
        ax[1, i].set_ylabel("Prediction")
    else:
        ax[0, i].axis('off')
        ax[1, i].axis('off')

plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=1.0)

today = ''.join(str(date.today()).split('-'))[2:]
plt.savefig('stroke_3DCNN_results_%s.png' % (today))



# left_lesion = np.where(left, np.array([250, 50, 0], dtype='uint8'), right)
# # left_result = right + left_lesion
# left_result = cv2.addWeighted(right, 1, left_lesion, 0.8, 0)
# # cv2.imshow('img', cv2.cvtColor(left_result, cv2.COLOR_BGR2RGB))

# middle_bg = np.copy(middle)
# middle_bg[(middle == 0).all(-1)] = [0, 0, 100]  # RGB; plt.imshow(middle_bg)

# mid_bg = [0, 0, 30] * np.ones(shape=[h, w_cut, 3], dtype='uint8')
# mid_bg = np.where(right, np.array([0, 0, 150], dtype='uint8'), middle)
# mid_lesion = np.where(middle, np.array([250, 50, 0], dtype='uint8'), right)
# # mid_lesion = np.where(middle, np.array([0, 250, 30], dtype='uint8'), np.array([0, 0, 150], dtype='uint8'))
# # mid_result = right + mid_lesion // 2
# mid_mask = cv2.addWeighted(mid_bg, 0.2, mid_lesion, 0.8, 0)
# mid_result = cv2.addWeighted(right, 1, mid_mask, 1, 0)
# cv2.imshow('img', cv2.cvtColor(mid_result, cv2.COLOR_BGR2RGB))

# # left_bg = np.where(right, np.array([0,0,200], dtype='uint8'), left); plt.imshow(left_bg)
# # left_lesion = cv2.cvtColor(cv2.bitwise_and(left, right), cv2.COLOR_BGR2LAB)
# left_lesion = cv2.bitwise_and(left, right); plt.imshow(left_lesion)
# left_lesion_color = cv2.applyColorMap(left_lesion, cv2.COLORMAP_HSV); plt.imshow(left_lesion_color)
# # left_lesion_color = np.where(left_lesion_color, np.array([0,180,255], dtype='uint8'), right)
# plt.imshow(left_lesion_color)
# img_bg = cv2.addWeighted(left_bg, 0.2, right, 1, 0)
# img_out = cv2.addWeighted(right, 1, left_lesion_color, 0.8, 0)
# plt.imshow(img_out)

# cv2.imshow('img', img_out)
# plt.imshow(img_out)
