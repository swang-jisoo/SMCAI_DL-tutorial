import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from datetime import date

# Please adjust the value of variables below
# path to the images
paths = glob.glob("C:\\Users\\swang\\PycharmProjects\\results - stroke\\3D CNN storke_result_20210405\\*.png")
nrow, ncol = 2, len(paths)  # 1st row=GT, 2nd row=Prediction
# file name to save
today = ''.join(str(date.today()).split('-'))[2:]
save_name = 'stroke_3DCNN_results_%s.png' % (today)

# Plot the GT/Prediction mask on the top of original images
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
plt.savefig(save_name)
