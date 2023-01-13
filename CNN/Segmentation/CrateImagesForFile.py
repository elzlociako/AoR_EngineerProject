import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def SaveFig(input, path, cmap=None):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(input, aspect='auto',cmap=cmap)
    plt.show()
    fig.savefig(path, format='png')

def load_axis_depth(top_dir):
    images_dataset = []
    for root, dirs, files in os.walk(top_dir):
        for name in files:
            img = np.array(np.load(os.path.join(root, name)))
#             print(name)
            np.array(images_dataset.append(img))
    return np.array(images_dataset)

def load_from_csv(series, data_type, row=0):
    dataset = []
    for n in range(series.shape[0]):
        if data_type=='rgb':
            img = np.array(Image.open(series.iloc[n,row]).convert('L'))
        if data_type=='depth':
            img = np.array(np.load(series.iloc[n,row]))
        np.array(dataset.append(img))
        
    return np.array(dataset)



def main(args=None):
    df = pd.read_csv('./files/data.csv') 
    df_INPUT_DEPTH = df[['depth_img_I', 'depth_img_II']]
    df_INPUT_RGB = df[['rgb_img_I', 'rgb_img_II']]

    RGBimg_begin = load_from_csv(df_INPUT_RGB, 'rgb' ,0)
    RGBimg_end = load_from_csv(df_INPUT_RGB, 'rgb' ,1)
    DEPTHimg_begin = load_from_csv(df_INPUT_DEPTH, 'depth',0)
    DEPTHimg_end = load_from_csv(df_INPUT_DEPTH, 'depth', 1)
    DEPTH_axis = load_axis_depth('./files/axis')

    img_nr = 66
    # SaveFig(RGBimg_begin[img_nr], 'Images/test1.png', cmap='gray')
    # SaveFig(DEPTHimg_begin[img_nr], 'Images/test2.png', cmap=None)
    # SaveFig(np.abs(DEPTHimg_begin[img_nr]-DEPTHimg_end[img_nr]), 'Images/test3.png', cmap=None)
    # SaveFig(DEPTH_axis[img_nr], 'Images/glebia55.png')
    test = np.load('/home/el_zlociako/Documents/AoR_CNN/Segmentacja/Images/test_seg.npy')
    elo = test.flatten()
    elo = elo[np.nonzero(elo)]
    print(elo)
    t = np.arange(elo.shape[0])
    plt.scatter(t, elo)
    plt.show()


if __name__ == '__main__':
    main()
