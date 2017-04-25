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
    features_name:

    min_Gain: float,optional(default = 0)
        used for pruning 
    prune_Criterion: string, optional(default = 'misclassificationRate')
        used for pruning. 

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
                 prune_Criterion = 'misclassificationRate',
                 min_Gain = 0,
                 notify = False,
                 min_samples_split = 2):
        # Representation: node and reference
        self.criterion = criterion
        self.featues_name = features_name
        self.categorical_predictors = categorical_predictors
        self.min_samples_split = min_samples_split

        self.min_Gain = min_Gain
        self.prune_Criterion = prune_Criterion
        self.notify = notify
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


    def plotTree(self,indent = ' ',verbose = False):
        """plot the obtained decision tree"""
        result_string = toString(self.rootNode,self.featues_name,self.categorical_predictors)
        if verbose:
            print(result_string)
        return result_string

    def prune(self,mergeNeighbors = False):
        # This function implement a sub-tree replacement post-pruning algorithm
        subtreeReplacement(self.rootNode,self.min_Gain,self.prune_Criterion,self.notify)
        if mergeNeighbors: # If mergeNeighbors is True, merge silbings with same class
            mergeNeighbors(self.rootNode)

def mergeNeighbors(node):   
    if node.children:
        for child in node.children:
            mergeNeighbors(child)
    # merge siblings with same label
    if node.children: # if node is not a leaf
        if node.cutPoint: # if cut type is numerical
            newChildren = []
            newChildren.append(node.children[0])
            cutPoints_ToRemove = []
            for i in range(len(node.children)-1):
                if ((not node.children[i].children) and (not node.children[i+1].children)) \
                    and  majorClass(node.children[i].dataset) == majorClass(node.children[i+1].dataset):
                 #if both these two children are leaves and their classes are 
                    newChildren[-1].dataset = node.children[i].dataset + node.children[i+1].dataset
                    cutPoints_ToRemove.append(node.cutPoint[i])
                else:
                    newChildren.append(node.children[i+1])
            for point in cutPoints_ToRemove:
                node.cutPoint.remove(point)
            node.children = newChildren
            if len(node.children) == 1: # if there is only one child left, this node should be pruned
                node.children = []
                node.cutPoint = []
        else: #binominal
            if ((not node.children[0].children) and (not node.children[1].children)) \
                    and  majorClass(node.children[0].dataset) == majorClass(node.children[0].dataset):
                node.children = []
                node.cutCategories = []
    return

        




def subtreeReplacement(node, minGain,pruneCriterion = 'entropy',notify = False):
    #determine the prunign criterion: "entropy", "gini" or "misclassificationRate"
    if pruneCriterion == 'entropy':
        evaluationFunction = entropy
    elif pruneCriterion == 'gini':
        evaluationFunction = gini
    else:
        evaluationFunction = misclassificationRate

    # recursive call for each branch
    log = False
    if node.children:
        for child in node.children:
            if child.children:
                # if this child is not a leaf, 
                subtreeReplacement(child,minGain,pruneCriterion,notify)
            if child.children:# if this child is still not a leaf
                break
            else:
                log = log or child.children 
        else:
            if not log:
            #The children of current node are all leaves
            # merge leaves(potentially)
                newScore = 0
                for child in node.children:
                    p = len(child.dataset)/len(node.dataset)
                    newScore += p*evaluationFunction(child.dataset)
                delta = evaluationFunction(node.dataset) - newScore
                if delta < minGain:
                    if notify: print('A branch was pruned: gain = %f'%delta)
                    node.children = []
             
def toString(node,features_name,categorical_predictors,indent= ' '):
    if not node.children:
        return indent + str(node.majorClass[0])
    else:
        if categorical_predictors[node.cutPredictor]:
            result_String = ''
            for i,child in enumerate(node.children):
                decision = indent + features_name[node.cutPredictor] +'=='+ str(node.cutCategories[i])
                branch = toString(child,features_name,categorical_predictors,indent +'\t\t')
                result_String = result_String + decision +'\n' + branch + '\n'
        else:#if splitting varible is continuous
            cutPoints = [-np.inf] + node.cutPoint + [np.inf]
            intervals = []
            result_String = ''
            for i,child in enumerate(node.children):
                interval = str(cutPoints[i]) + ' to '  + str(cutPoints[i+1])
                intervals.append(interval)
                decision = indent + features_name[node.cutPredictor] + ' in ' + interval
                branch = toString(child,features_name,categorical_predictors,indent +'\t\t')
                result_String = result_String + decision +'\n' + branch + '\n'
        return result_String

def uniqueCount(rows):
    results = {}
    ResponseVar = [row[-1] for row in rows]
    ResponseVar_set = set(ResponseVar)
    for ele in ResponseVar_set:
        results[ele] = ResponseVar.count(ele)
    return results

def misclassificationRate(rows):
    mClass,count = majorClass(rows)
    return 1 - count/len(rows)
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

def divideSet(rows, column, categorical = 0,mergeNeighbors = False):
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

        if mergeNeighbors:# merge adjacent subsets with have same major class
            newLists = []
            newLists.append(lists[0])
            cutPoints_Toremove = []
            
            for i in range(len(lists)-1):
                if majorClass(lists[i]) == majorClass(lists[i+1]):
                    cutPoints_Toremove.append(cutPoints[i])
                    newLists[-1] = newLists[-1] + lists[i+1]
                    print('two subsets was merged,class1:%s,class2:%s'%(majorClass(lists[i]),majorClass(lists[i])))
                else:
                    newLists.append(lists[i+1])
            for point in cutPoints_Toremove:
                cutPoints.remove(point)
            lists = newLists

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