"""This module implements a multi-way decision tree. 

1) The difference of this DT from popular DTs(e.g. C4.5, CART) 
pertain to the splitting function for numerical variables. 

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

class Node:
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
        # Representation: node and reference
        self.children = []
        self.cutPredictor = None  
        self.cutType = None
        self.cutPoint = []
        self.cutCategories = []
    
    def setData(self,rows):
        self.dataset = rows
    def setCut(self,cutPredictor,cutType,cutPoint,cutCategories,children):
        self.cutPredictor = cutPredictor
        self.cutType = cutType
        self.cutPoint = cutPoint
        self.cutCategories = cutCategories
        self.children = children

class DecisionTree:
    '''Multi-way tree classifier


    Parameters: 
    _______________
    criterion: string, optional(default = "entropy")
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
                 criterion = 'entropy',
                 categorical_predictors = None,
                 min_samples_split = 2):
        # Representation: node and reference
        self.criterion = criterion
        self.categorical_predictors = categorical_predictors
        self.min_samples_split = min_samples_split

        self.rootNode = None
       
    def fit(self,trainData):
        '''Grows and then returns a decision tree
        ---|----      |----      |---
        No.|author    |date      |action
        ---|------    |------    |----
        1  | Fan Feng | 2017/4/20|Create 

        Inputs:
        ----------------------------------
        trainData:
             a n*m matrix, and the last column is target/result variable
             e.g. 
              [[1,2,3,'T'],
               [1,2,4,'F'],
               [2,1,3,'T']]

        '''
        # Determine output settings
        n_samples,n_features = len(trainData),len(trainData[0])-1
        # Determine the evaluationFunction
        if self.criterion =="gini":
            evaluationFunction = gini
        else:
            evaluationFunction = entropy
        # Determine the categorical_predictors
        if self.categorical_predictors == None:
            self.categorical_predictors = [0]*n_features 

        root_Node = Node()
        root_Node.setData(trainData)
        self.rootNode = root_Node
        DFSGrowTree(root_Node,self.min_samples_split,self.categorical_predictors,evaluationFunction)
     
def uniqueCount(rows):
    results = {}
    ResponseVar = [row[-1] for row in rows]
    ResponseVar_set = set(ResponseVar)
    for ele in ResponseVar_set:
        results[ele] = ResponseVar.count(ele)
    return results

def entropy(rows):
    from math import log
    log2 = lambda x:log(x)/log(2)
    results = uniqueCount(rows)

    entr = 0.0
    for r in results:
        p = float(results[r])/len(rows)
        entr -= p*log2(p)
    return entr

def gini(rows):
    # $ gini = \sum_{k=1}^{K}{P_{k}*(1-P_{k})} $
    results = uniqueCount(rows)

    G = 0.0
    for r in results:
        p = float(results[r])/len(rows)
        G += p*(1-p)
    return G

def divideSet(rows, column, categorical = 0):
    cutPoint = []
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
                cutPoint.append((rows_sorted[i-1][column]+rows_sorted[i][column])/2)
                lists.append(l)
                l = []
            l.append(rows_sorted[i])
        lists.append(l)
    return lists,cutPoint
def DFSGrowTree(current_Node,min_samples_split,categorical_predictors,evaluationFunction = entropy):
    # This function grow a decision tree in a recursive manner
    rows = current_Node.dataset

    if len(rows)<=min_samples_split:return 
    currentScore = evaluationFunction(rows)
    #Step1: find the best cut predictors for the current nodes
    bestGain = 0
    bestAttribute = 0
    bestCutPoint = []
    bestSets = []

    n_samples,n_features = len(rows),len(rows[0])-1

    for column in range(0,n_features):
        (sets,cutPoint) = divideSet(rows,column,categorical_predictors[column])
        #Gain: Entropy or Gini
        newScore = 0
        for set in sets:
            p = float(len(set))/len(rows)
            newScore += p*evaluationFunction(set)
        gain = currentScore - newScore
        if gain > bestGain:
            bestGain = gain
            bestAttribute = column
            bestCutPoint = cutPoint
            bestSets = sets
    #Step2: grow each branch recursively. 
    if bestGain > 0: # If the splitting process do not stop.
        # if bestAttribute is categorical
        cutCategories = set([row[bestAttribute] for row in rows]) if categorical_predictors[bestAttribute] else []
        children = []
        for set in bestSets:
            node_Temp = Node()
            node_Temp.setData(set)
            children.append(node_Temp)
        current_Node.setCut(bestAttribute,categorical_predictors[bestAttribute],bestCutPoint,cutCategories,children)
        for child in current_Node.children:
            DFSGrowTree(child,min_samples_split,categorical_predictors,evaluationFunction)
        return
    else:
        return

