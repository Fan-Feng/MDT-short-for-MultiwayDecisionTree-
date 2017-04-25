from MultiwayDecisionTree import DecisionTree
from discretization import MDLP_Discretizer
import pandas as pd
import numpy as np

'''
rows =  [[1,2,2,'T'],
        [1,2,7,'F'],
        [2,1,3,'T'],
         [2,1,5,'F']]
'''
data = pd.read_csv('testData.csv', na_values = '?')
data = data.fillna(method = 'bfill',axis = 0)
data = data.fillna(0)

rows = np.array(data)
DTClassifier = DecisionTree(categorical_predictors = [1,1,1,0,0,1,1,1,0] + [1]*23 +[0,0,0] + [1,1,1],
                            features_name = ['features%s'%i for i in range(38)],
                            min_Gain = 0.4)
DTClassifier.fit(rows)
bef = DTClassifier.plotTree()
DTClassifier.prune(mergeNeighbors = True)
print("Decision tree successfully fitted")
after = DTClassifier.plotTree()