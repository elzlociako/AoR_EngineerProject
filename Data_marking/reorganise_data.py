import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

rng = default_rng()

dataset = pd.read_csv('/home/el_zlociako/Documents/Praca_inzynierska/Dataset/files_ArUco/data_ArUco.csv')
P_VAL = 0.04
REMOVED_NUMBER = int(dataset.shape[0] * P_VAL)

RandomSamples = rng.choice(dataset.shape[0], REMOVED_NUMBER, replace=False).tolist()
RandomSamples.sort()
print(RandomSamples)

Validation = dataset.loc[RandomSamples]
Train = dataset.drop(RandomSamples)

print(Train.shape)
print(Validation.shape)

Validation.to_csv('/home/el_zlociako/Documents/Praca_inzynierska/Dataset/files_ArUco/data_VAL.csv', index=False)
Train.to_csv('/home/el_zlociako/Documents/Praca_inzynierska/Dataset/files_ArUco/data_TRA.csv', index=False)

