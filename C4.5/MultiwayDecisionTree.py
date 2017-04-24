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
import pandas as pd
import numpy as np
from discretization import MDLP_Discretizer


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

        self.majorClass = []
    
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

    This multiway decision tree classifier:
    Splitting:
    1) continuous variables:
       local entropy-based discretization
    2) categorical variables:
        mutliway splitting used in C4.5 

    Pruning:
    Each variable will be used at most once, hence it's unnecessary 
    to prune. 

    Parameters: 
    _______________
    criterion: string, optional(default = "entropy")
        The function to measure the quality of a split. Supportedd criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    categorical_predictors: boolean list, optional (default = None)
        categorical_predictors is a boolean list of length n_features.



    Example
    _______________
    >>> from MultiwayDecisionTree import DecisionTree
    >>>DT = DecisionTree(categorical_predictors = [1,1,1])
    >> DT.fit(trainData)
    '''
    def __init__(self,
                 criterion = 'entropy',
                 categorical_predictors = None,
                 features_name = None,
                 min_samples_split = 2):
        # Representation: node and reference
        self.criterion = criterion
        self.featues_name = features_name
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
        if self.categorical_predictors is None:
            self.categorical_predictors = [0]*n_features 

        root_Node = Node()
        root_Node.setData(trainData)
        root_Node.majorClass = majorClass(trainData)
        self.rootNode = root_Node
        DFSGrowTree(root_Node,self.min_samples_split,self.categorical_predictors,evaluationFunction)


    def plotTree(self,indent = ' '):
        """plot the obtained decision tree"""
        result_string = toString(self.rootNode,self.featues_name,self.categorical_predictors)
        print(result_string)
        return result_string

def toString(node,features_name,categorical_predictors,indent= ' '):
    if not node.children:
        return str(node.majorClass[0])
    else:
        if categorical_predictors[node.cutPredictor]:
            result_String = ''
            for i,child in enumerate(node.children):
                decision = features_name[node.cutPredictor] +'=='+ str(node.cutCategories[i])
                branch = indent + toString(child,features_name,categorical_predictors,indent +'\t\t')
                result_String = result_String + decision +'\n' + branch + '\n'
        else:#if splitting varible is continuous
            cutPoints = [-np.inf] + node.cutPoint + [np.inf]
            intervals = []
            result_String = ''
            for i,child in enumerate(node.children):
                interval = str(cutPoints[i]) + ' to '  + str(cutPoints[i+1])
                intervals.append(interval)
                decision = features_name[node.cutPredictor] + ' in ' + interval
                branch = indent + toString(child,features_name,categorical_predictors,indent +'\t\t')
                result_String = result_String + decision +'\n' + branch + '\n'
        return result_String

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
    cutPoints = []
    if categorical: # the splitting variable is categorical
        values = list(set([row[column] for row in rows]))
        lists = []
        for value in values:
            l = [row for row in rows if row[column] == value]
            lists.append(l)
    else: # the splitting variable is numerical
        # local entropy-based discretization 
        data = pd.DataFrame(rows,columns = ['feature%s'%i for i in range(len(rows[0])-1)] +['label'])

        discretizer = MDLP_Discretizer(data,class_label = 'label',\
                               out_path_data = 'result.csv',out_path_bins = 'result_bins.csv',features = ['feature%s'%column])
        lists = [np.array(ele) for ele in discretizer.subsets]
        cutPoints = discretizer.cutPoints
    return lists,cutPoints

def majorClass(s):
    # return major class in Set s
    labels = [row[-1] for row in s]
    values = set(labels)
    major = labels[0]
    count = 1
    for value in values:
        temp = labels.count(value)
        if temp > count:
            count = temp
            major = value
    return major,count

def DFSGrowTree(current_Node,min_samples_split,categorical_predictors,evaluationFunction = entropy,features_left= None):
    # This function grow a decision tree in a recursive manner
    rows = current_Node.dataset
    if features_left is None:
        features_left = range(len(rows[0])-1)

    if len(rows)<=min_samples_split:return 
    currentScore = evaluationFunction(rows)
    #Step1: find the best cut predictors for the current nodes
    bestGain = 0
    bestAttribute = 0
    bestCutPoint = []
    bestSets = []

    n_samples,n_features = len(rows),len(rows[0])-1

    for column in features_left:
        (sets,cutPoint) = divideSet(rows,column,categorical_predictors[column])
        #Gain: Entropy or Gini
        newScore = 0
        for s in sets:
            p = float(len(s))/len(rows)
            newScore += p*evaluationFunction(s)
        gain = currentScore - newScore
        if gain > bestGain:
            bestGain = gain
            bestAttribute = column
            bestCutPoint = cutPoint
            bestSets = sets
    #Step2: grow each branch recursively. 
    if bestGain > 0: # If the splitting process do not stop.

        # exclude this feature(Attribute) from feature_left
        features_left = list(features_left)
        features_left.remove(bestAttribute)

        # if bestAttribute is categorical
        cutCategories = list(set([row[bestAttribute] for row in rows])) if categorical_predictors[bestAttribute] else []
        children = []
        for s in bestSets:
            node_Temp = Node()
            node_Temp.setData(s)
            node_Temp.majorClass = majorClass(s)
            children.append(node_Temp)
        current_Node.setCut(bestAttribute,categorical_predictors[bestAttribute],bestCutPoint,cutCategories,children)
        for child in current_Node.children:
            DFSGrowTree(child,min_samples_split,categorical_predictors,evaluationFunction,features_left)
        return
    else:
        return

    