import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from datetime import date

# Please adjust the value of variables below
# path to the images
paths = glob.glob("C:\\Users\\swang\\PycharmProjects\\results - stroke\\3D CNN storke_result_20210405\\*.png")

# ytitle = ["DWI", "Ground Truth", "AI Prediction"]
ytitle = ["DWI", "ADC", "Ground Truth", "AI Prediction"]
nrow, ncol = len(ytitle), len(paths)

# file name to save
today = ''.join(str(date.today()).split('-'))[2:]
save_name = 'stroke_3DCNN_results_%s.png' % (today)


# Plot the GT/Prediction mask on the top of original images
fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow))

for i in range(len(paths)):
    img = cv2.imread(paths[i], cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    w_cut = w // nrow
    # gt, seg, dwi = img[:, :w_cut, :], img[:, w_cut:w_cut*2, :], img[:, w_cut*2:w_cut*3, :]
    gt, seg, dwi, adc = img[:, :w_cut, :], img[:, w_cut:w_cut*2, :], img[:, w_cut*2:w_cut*3, :], img[:, w_cut*3:, :]

    # GT
    gt_mask = np.copy(gt)
    gt_mask[(gt > 0).all(-1)] = [128, 0, 0]
    gt_result = cv2.addWeighted(dwi, 1, gt_mask, 0.4, 0)

    # Prediction
    seg_mask = cv2.cvtColor(cv2.applyColorMap(seg, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    seg_mask[(seg_mask == [0, 0, 128]).all(-1)] = [0, 0, 0]
    seg_result = cv2.addWeighted(dwi, 1, seg_mask, 0.4, 0)

    # ximg = [dwi, gt_result, seg_result]
    ximg = [dwi, adc, gt_result, seg_result]
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
plt.savefig(save_name)
