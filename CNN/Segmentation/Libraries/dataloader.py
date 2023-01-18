import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid

import os
from PIL import Image

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        self.ResizeData = transforms.Resize([256, 320], interpolation=transforms.InterpolationMode.NEAREST)
        self.RGBtoGRAY = transforms.Grayscale(num_output_channels=1)


    def __load_from_csv(self, root_dir, series, data_type, row=0):
        dataset = []
        for n in range(series.shape[0]):
            if data_type=='rgb':
                img = np.array(Image.open(root_dir+series.iloc[n,row]))
            if data_type=='depth':
                img = np.array(np.load(root_dir+series.iloc[n,row]))
            np.array(dataset.append(img))
            
        return np.array(dataset)

    def __TransformImage(self, img,div_val,reshape=False,transform=False):
        if reshape == True:
            img = img.reshape((img.shape[0], img.shape[1], img.shape[2], 1))
                        
        img = img.transpose(0,3,1,2)
        img = torch.FloatTensor(img).div(div_val)
        
        return img
    
    def __load_axis_depth(self, top_dir):
        images_dataset = []
        for root, dirs, files in os.walk(top_dir):
            for name in files:
                img = np.array(np.load(os.path.join(root, name)))
                np.array(images_dataset.append(img))
        return np.array(images_dataset)

    def load(self, DATASET_ROOTDIR, csv_path, depth_dir, RorS):
        df = pd.read_csv(f'{DATASET_ROOTDIR}{csv_path}') 
        df_INPUT_DEPTH = df[['depth_img_I', 'depth_img_II']]
        df_INPUT_RGB = df[['rgb_img_I', 'rgb_img_II']]
        df_OUTPUT = df[['x1','y1','z1','x2','y2','z2']]

        RGBimg_begin = self.__load_from_csv(DATASET_ROOTDIR, df_INPUT_RGB, 'rgb' ,0)
        DEPTHimg_begin = self.__load_from_csv(DATASET_ROOTDIR, df_INPUT_DEPTH, 'depth',0)
        DEPTHimg_end = self.__load_from_csv(DATASET_ROOTDIR, df_INPUT_DEPTH, 'depth', 1)
        DEPTH_axis = self.__load_axis_depth(f'{DATASET_ROOTDIR}{depth_dir}')

        # Taking first rgb image
        rgb_in = torch.IntTensor(RGBimg_begin.transpose(0,3,1,2))
        gray_in = self.RGBtoGRAY(rgb_in).div(255)
        # plt.imshow(gray_in[0].numpy().transpose(1,2,0),cmap='gray')

        # Taking depth beginning state of the movement
        depthBeg_in = self.__TransformImage(DEPTHimg_begin,5500,reshape=True)

        # Taking depth end state of the movement
        # depthEnd_in = NormImage(DEPTHimg_end,5500,reshape=True)

        # Taking depth difference between movements
        depthDiff_in = self.__TransformImage(abs(DEPTHimg_begin - DEPTHimg_end),5500,reshape=True)

        # Taking outputs
        if(RorS=='S'):
            axis_out = self.ResizeData(self.__TransformImage(DEPTH_axis, 1, reshape=True))
        if(RorS=='R'):
            axis_out = torch.Tensor(df_OUTPUT.values)

        # Connected input for model
        RGBD_in = self.ResizeData(torch.cat((gray_in, depthBeg_in, depthDiff_in),axis=1))

        return [RGBD_in, axis_out]

