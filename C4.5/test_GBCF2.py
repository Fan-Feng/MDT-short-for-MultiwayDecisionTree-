import pandas as pd
import numpy as np
import os
# user-defined module
import MultiwayDecisionTree as MDT
import sequencingControl as sc

# read trainY of GBCF2
os.chdir(r'C:\Users\lenovo\Desktop\C4dot5')
trainX = pd.read_csv(r'.\GBCF2\originData\trainX.csv',parse_dates = [0,3])
trainY = pd.read_csv(r'.\GBCF2\originData\trainY.csv',parse_dates = [0])
trainY.columns = ['timestamp','data']
trainY = trainY.set_index('timestamp')
trainY.head()
sc.plot_heatMap(trainY.reset_index(),'{}.jpg'.format(1))