import numpy as np
import cv2

img_size = (256, 256)

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [224, 224, 192]]

y_onehot = np.zeros((img_size[0], img_size[1], nclass), dtype=int)

for j in range(nclass):
    A = VOC_COLORMAP[j]
    A = [A[2], A[1], A[0]]
    y_onehot[:, :, j] = cv2.inRange(y, np.array(A), np.array(A))
    # cv2.imshow("test", y_onehot[:, :, j].astype(float))
    # cv2.waitKey(0)