import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os

def Rename(rootdir, newdir, name, type, couter_beg):

    # os.mkdir(newdir)

    counter = couter_beg
    for subdir, dirs, images in os.walk(rootdir):

        for old_img in images:
            os.rename(os.path.join(subdir, old_img), os.path.join(newdir, f'{name}{str(counter).zfill(5)}.{type}'))
            counter+=1

dataset = pd.read_csv('files/data.csv')

for i in range(dataset.shape[0]):
    dataset.loc[i, 'nr'] = i

dir = '/home/el_zlociako/Documents/AoR_CNN/Segmentacja/rgb_img_I'
newdir = '/home/el_zlociako/Documents/AoR_CNN/Segmentacja/rgb_img_I/rgb_img_I_NEW'
Rename(dir, newdir,'R', 'png', 120)
