"""
This module implements a multi-way decision tree. 
1) The difference of this DT from 
popular DTs(e.g. C4.5, CART) pertain to the splitting function for numerical variables. 

The splitting function is quite similar to OneR algorithm, as following
X | 1 2 3 4 5 6         X | 1 2 ||3 4|| 5 6
--|------------  ==>    --|------------
Y | T T F F T T         Y | T T ||F F ||T T
This algorithm places breakpoints wherever the class changes. 

2) The splitting function for categorical variables is same as the function used in C4.5 algorithm.

3) Pruning function

"""
# Author: Fan Feng, Email: 18817598306@163.com
# License: BSD 3 clause

import collections

class node:
    '''Tree node
    Parameters
    _______________
    children: a list of nodes
    cutPredictor: string, cutPredictor
    cutType: string, "categorical" or "numerical"
    cutPoint: applicable when cutType is "numerical"
    cutCategories: applicable when cutType is "categorical"
    '''

    
    def __init__(self):
        # Representation�� node and reference
        self.children = []
        self.cutPredictor = []
        self.cutType = []
        self.cutPoint = []
        self.cutCategories = []



class decisionTree:
    '''Multi-way tree classifier


    Parameters: 
    _______________
    criterion�� string, optional(default = "entropy")
        The function to measure the quality of a split. Supportedd criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    categorical_predictors: boolean list, optional (default = None)
        categorical_predictors is a boolean list of length n_features.



    Example
    _______________
    >>> from
    >>>
    '''
    def __init__(self,
                 criterion = 'gini',
                 categorical_predictors = None,
                 min_samples_split = 2):
        # Representation�� node and reference
        self.criterion = criterion
        self.categorical_predictors = categorical_predictors
        self.min_samples_split = min_samples_split

        self.rootNode = None
       
    def uniqueCount(self,rows):
        results = {}
        ResponseVar = [row[-1] for row in rows]
        ResponseVar_set = set(ResponseVar)
        for ele in ResponseVar_set:
            results[ele] = ResponseVar.count(ele)
        return results

    def entropy(self,rows):
        from math import log
        log2 = lambda x:log(x)/log(2)
        results = uniqueCount(rows)

        entr = 0.0
        for r in results:
            p = float(results[r])/len(rows)
            entr -= p*log2(p)
        return entr

    def gini(self,rows):
        # $ gini = \sum_{k=1}^{K}{P_{k}*(1-P_{k})} $
        results = uniqueCount(rows)

        G = 0.0
        for r in results:
            p = float(results[r])/len(rows)
            G += p*(1-p)
        return G

    def divideSet(self,rows, column, categorical = 0):
        if categorical: # the splitting variable is categorical
            values = set([row[column] for row in rows])
            lists = []
            for value in values:
                l = [row for row in rows if row[column] == value]
                lists.append(l)
        else: # the splitting variable is categorical
            # sort rows according to column, 
            sort_idx = sorted(range(len(rows)),key = lambda k:rows[k][column])
            rows_sorted = [rows[ele] for ele in sort_idx]

            lists = []
            l = []
            l.append(rows_sorted[0])
            for i in range(1,len(rows)):
                if rows_sorted[i-1][-1] != rows_sorted[i][-1]:
                    lists.append(l)
                    l = []
                l.append(rows_sorted[i])
            lists.append(l)
        return lists

    def fit(self,trainData):
        '''Grows and then returns a decision tree
        ---|----      |----      |---
        No.|author    |date      |action
        ---|------    |------    |----
        1  | Fan Feng | 2017/4/20|Create 

        Inputs:
        ----------------------------------
        trainData��
             a n*m matrix, and the last column is target/result variable
             e.g. 
              [[1,2,3,'T'],
               [1,2,4,'F'],
               [2,1,3,'T']]

        '''
        # Determine the evaluationFunction
        if self.criterion =="gini":
            evaluationFunction = gini
        else:
            evaluationFunction = entropy
        # Determine output settings
        n_samples,n_features = len(trainData),len(trainData[0])-1


    def DFSGrowTree(self,rows,evaluationFunction = entropy):
        # This function grow a decision tree in a recursive manner
        if len(rows)<=self.min_samples_split:return node()
        currentScore = evaluationFunction(rows)
        #Step1: find the best cut predictors for the current nodes
        bestGain = 0
        bestAttribute = None
        bestSets = None

        n_samples,n_features = len(rows),len(rows)-1
        for feature in range(0,n_features):
            columnValues = [row[feature] for row in rows]
 
            if self.categorical_predictors[feature]: # if this feature is categorical



        #Step2: grow each branch recursively. 


        return 
    
    def predict():


