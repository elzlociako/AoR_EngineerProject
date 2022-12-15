import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

dataset = pd.read_csv('files/data.csv')

for i in range(dataset.shape[0]):
    dataset.loc[i, 'nr'] = i
dataset.to_csv("files/data.csv", index=False)
di = np.load('/home/el_zlociako/Documents/AoR_CNN/Segmentacja/files/axis/AX00000.npy')
path = np.load('/home/el_zlociako/Documents/AoR_CNN/Segmentacja/files/images/depth_img_II/D00000.npy')

print(path)
print(di)

plt.figure()
plt.imshow(path)
plt.figure()
plt.imshow(di*1000)
plt.show()
# img = np.zeros([480,640], float)
# tmp_img = np.zeros([480,640], float)
# depth_img = np.load(dataset['depth_img_II'].iloc[1])
# cv.circle(tmp_img, (int(10), int(340)), radius=5, color=230, thickness=-1)
# if np.sum(tmp_img*depth_img):
#     img[np.where(tmp_img!=0)] = tmp_img[np.where(tmp_img!=0)]
# plt.imshow(img)
# plt.show()