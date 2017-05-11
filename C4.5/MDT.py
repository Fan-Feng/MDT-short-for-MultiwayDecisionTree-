#usage
# python MDT.py -d dataset.csv 

from MultiwayDecisionTree import DecisionTree
from discretization import MDLP_Discretizer
import argparse
import pandas as pd
import numpy as np
import sys
'''
rows =  [[1,2,2,'T'],
        [1,2,7,'F'],
        [2,1,3,'T'],
         [2,1,5,'F']]
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--data',
                        type =str,
                         dest = 'dataFile',
                         help = 'fileName of dataset',
                         default = None)
    parser.add_argument('-o','--output',
                         dest = 'output',
                         help = 'output file Name, .txt',
                         default = None)
    parser.add_argument('-c','--categorical_predictors',
                        nargs = '+',
                         dest = 'categorical_predictors',
                         help = 'categorical_predictors,e.g. [1,1,0,1]',
                         default = None)
    parser.add_argument('-g','--min_Gain',
                         dest = 'min_Gain',
                         help = 'minimum gain， used in control the size of the tree',
                         default = None)

    args = parser.parse_args()
    data = None
    if not args.dataFile:
        dataFile = sys.stdin
    elif args.dataFile:
        data = pd.read_csv(args.dataFile)
    else:
        print('No dataset fileName specified, system will exit\n')
        sys.exit('System will exit')

    data = data.fillna(method = 'bfill',axis = 0)
    data = data.fillna(0)
    rows = np.array(data)

    categorical_predictors = args.categorical_predictors
    min_Gaim = args.min_Gain
    features_Name = list(data.columns)[:-1]

    DTClassifier = DecisionTree(categorical_predictors = categorical_predictors,
                            features_name = features_Name,
                            min_Gain = 0.4)
    #fit tree
    DTClassifier.fit(rows)
    bef = DTClassifier.plotTree()
    DTClassifier.prune(mergeNeighbors = True)
    print("Decision tree successfully fitted")
    after = DTClassifier.plotTree(verbose = True)

    output = args.output
    f = open(output,'w')
    f.write(after)
    f.close