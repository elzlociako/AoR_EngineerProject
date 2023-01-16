import numpy as np
import pandas as pd

ROOT_DIR = '/home/el_zlociako/Documents/Praca_inzynierska/Dataset/'
pd = pd.read_csv(f'{ROOT_DIR}files_ArUco/data_ArUco.csv')

for row in range(pd.shape[0]):
    pd.loc[row, 'K0'] = 570
    pd.loc[row, 'K1'] = 0
    pd.loc[row, 'K2'] = 320
    pd.loc[row, 'K3'] = 0
    pd.loc[row, 'K4'] = 570
    pd.loc[row, 'K5'] = 240
    pd.loc[row, 'K6'] = 0
    pd.loc[row, 'K7'] = 0
    pd.loc[row, 'K8'] = 1

pd.to_csv("/home/el_zlociako/Documents/Praca_inzynierska/Dataset/files_ArUco/data_ArUco.csv", index=False)