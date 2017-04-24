import pandas as pd
import numpy as np
import os
# user-defined module
import MultiwayDecisionTree as MDT
import sequencingControl as sc

# read trainY of GBCF2
trainData = pd.read_csv('trainData.csv')
trainData = np.array(trainData)

dtClassifier = MDT.DecisionTree(features_name = ['month','day','time','weekendOrHoliday','temp'],
                                categorical_predictors = [0,0,0,1,0])
dtClassifier.fit(trainData)