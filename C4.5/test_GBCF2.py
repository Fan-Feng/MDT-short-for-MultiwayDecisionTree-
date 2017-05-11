import pandas as pd
import numpy as np
import os
# user-defined module
import MultiwayDecisionTree as MDT
import sequencingControl as sc

# read trainY of GBCF2
data = pd.read_csv('dataset.csv')
data = data.fillna(method = 'bfill',axis = 0)
data = data.fillna(0)
rows = np.array(data)

DTClassifier = MDT.DecisionTree(categorical_predictors = [0,0,0,1,0,0],
                            features_name = ['month','day','time','weekendOrHoliday','temp','dayOfYear'],
                            min_Gain = 0.4)

DTClassifier.fit(rows)
DTClassifier.prune()
a = DTClassifier.plotTree()
print('a')