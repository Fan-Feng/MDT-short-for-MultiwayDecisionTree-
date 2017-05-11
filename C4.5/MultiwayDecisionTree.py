"""This module implements a multi-way decision tree. 

------------

Author    |date      | Action
----------+----------+------
Fan Feng  |2017/4/19 | Create
Fan Feng  |2017/5/11 | Modify the pattern of this module

-------------
1) The difference of this DT from popular DTs(e.g. C4.5, CART) 
pertain to the splitting function for numerical variables. 

The splitting function is Entropy-based discretization.

X | 1 2 3 4 5 6         X | 1 2 ||3 4|| 5 6
--|------------  ==>    --|------------
Y | T T F F T T         Y | T T ||F F ||T T

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
        #Representation: node and reference
        self.__children = None
        self.__cutPredictor = None  
        self.__cutType = None
        self.__cutPoint = None
        self.__cutCategories = None
        self.__majorClass = None
        self.__dataset = None

    def setMajorClass(self,majorClass):
        self.__majorClass = majorClass
    def getMajorClass(self):
        return self.__majorClass
    def setData(self,rows):
        self.__dataset = rows
    def getData(self):
        return self.__dataset

    def setCut(self,cutPredictor,cutType,cutPoint,cutCategories):
        self.__cutPredictor = cutPredictor
        self.__cutType = cutType
        self.__cutPoint = cutPoint
        self.__cutCategories = cutCategories
    def getCutPredictor(self):
        return self.__cutPredictor
    def getCutType(self):
        return self.__cutType
    def setCutPoint(self,cutPoint):
        self.__cutPoint = cutPoint
    def getCutPoint(self):
        return self.__cutPoint
    def getCutCategories(self):
        return self.__cutCategories

    def setChildren(self,children):
        self.__children = children
    def getChildren(self):
        return self.__children

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
        self.__criterion = criterion
        self.__featues_name = features_name
        self.__categorical_predictors = categorical_predictors
        self.__min_samples_split = min_samples_split
        self.__min_Gain = min_Gain
        self.__prune_Criterion = prune_Criterion
        self.__notify = notify
        self.__rootNode = None      
    def fit(self,trainData):
        '''Grows and then returns a decision tree

        Inputs:
        ----------------------------------
        trainData:
             a n*m matrix, and the last column is target/result variable
             e.g. 
              [[1,2,3,'T'],
               [1,2,4,'F'],
               [2,1,3,'T']]
        ---|----      |----      |---
        No.|author    |date      |action
        ---|------    |------    |----
        1  | Fan Feng | 2017/4/20|Create 
        '''
        # Determine output settings
        n_samples,n_features = len(trainData),len(trainData[0])-1
        # Determine the evaluationFunction
        if self.__criterion =="gini":
            evaluationFunction = gini
        else:
            evaluationFunction = entropy
        # Determine the categorical_predictors
        if self.__categorical_predictors is None:
            self.__categorical_predictors = [0]*n_features 

        root_Node = Node()
        root_Node.setData(trainData)
        root_Node.majorClass = majorClass(trainData)

        self.__rootNode = root_Node
        DFSGrowTree(root_Node,self.__min_samples_split,self.__categorical_predictors,evaluationFunction)


    def plotTree(self,indent = ' ',verbose = False):
        """plot the obtained decision tree"""
        result_string = toString(self.__rootNode,self.__featues_name,self.__categorical_predictors)
        if verbose:
            print(result_string)
        return result_string

    def prune(self,mergeNeighbors = False):
        # This function implement a sub-tree replacement post-pruning algorithm
        subtreeReplacement(self.__rootNode,self.__min_Gain,self.__prune_Criterion,self.__notify)
        if mergeNeighbors: # If mergeNeighbors is True, merge silbings with same class
            mergeNeighbors_Fun(self.__rootNode)

## standalone function
def mergeNeighbors_Fun(node):   
    children = node.getChildren()
    if children:
        for child in children:
            mergeNeighbors_Fun(child)

    # merge siblings with same label
    if children: # if node is not a leaf
        cutPoint = node.getCutPoint()
        if cutPoint: # if cut type is numerical
            newChildren = []
            newChildren.append(children[0])
            cutPoints_ToRemove = []
            for i in range(len(children)-1):
                if ((not children[i].getChildren()) and (not children[i+1].getChildren() )) \
                    and  majorClass(children[i].getData())[0] == majorClass(children[i+1].getData())[0]:
                 #if both these two children are leaves and their classes are the same
                    if isinstance(children[i].getData(),list) and isinstance(children[i+1].getData(),list):
                        newChildren[-1].setData(children[i].getData() + children[i+1].getData)
                    elif isinstance(children[i].getData(),np.ndarray) and isinstance(children[i].getData(),np.ndarray):
                        newChildren[-1].setData(np.concatenate((children[i].getData(),children[i+1].getData())))
                    cutPoints_ToRemove.append(cutPoint[i])
                else:
                    newChildren.append(children[i+1])

            for point in cutPoints_ToRemove:
                cutPoint.remove(point)

            node.setCutPoint(cutPoint)
            node.setChildren(newChildren)

            if len(newChildren) == 1: # if there is only one child left, this node should be pruned
                node.setChildren([])
                node.setCutPoint([])
        else: #binominal
            if ((not node.children[0].children) and (not node.children[1].children)) \
                    and  majorClass(node.children[0].dataset)[0] == majorClass(node.children[1].dataset)[0]:
                node.setChildren([])
                node.setCutPoint([])
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
    children = node.getChildren()
    if children:
        for child in children:
            if child.getChildren():
                # if this child is not a leaf, 
                subtreeReplacement(child,minGain,pruneCriterion,notify)
            if child.getChildren():# if this child is still not a leaf
                break
            else:
                log = log or child.getChildren()
        else:
            if not log:
            #The children of current node are all leaves
            # merge leaves(potentially)
                newScore = 0
                children = node.getChildren()
                for child in children:
                    p = len(child.getData())/len(node.getData())
                    newScore += p*evaluationFunction(child.getData())
                delta = evaluationFunction(node.getData()) - newScore
                if delta < minGain:
                    if notify: print('A branch was pruned: gain = %f'%delta)
                    node.setChildren([])
             
def toString(node,features_name,categorical_predictors,indent= ' '):
    children = node.getChildren()
    if not children:
        return indent + str(node.getMajorClass()[0])
    else:

        if categorical_predictors[node.getCutPredictor()]:
            result_String = ''
            for i,child in enumerate(children):
                decision = indent + features_name[node.getCutPredictor()] +'=='+ str(node.getCutCategories()[i])
                branch = toString(child,features_name,categorical_predictors,indent +'\t\t')
                result_String = result_String + decision +'\n' + branch + '\n'
        else:#if splitting varible is continuous
            cutPoints = [-np.inf] + node.getCutPoint() + [np.inf]
            intervals = []
            result_String = ''
            for i,child in enumerate(children):
                interval = str(cutPoints[i]) + ' to '  + str(cutPoints[i+1])
                intervals.append(interval)
                decision = indent + features_name[node.getCutPredictor()] + ' in ' + interval
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
                if majorClass(lists[i])[0] == majorClass(lists[i+1])[0]:
                    cutPoints_Toremove.append(cutPoints[i])
                    newLists[-1] = newLists[-1] + lists[i+1]
                    print('two subsets was merged,class1:%s,class2:%s'%(majorClass(lists[i])[0],majorClass(lists[i])[0]))
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
    rows = current_Node.getData()
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
            node_Temp.setMajorClass(majorClass(s))
            children.append(node_Temp)
        current_Node.setCut(bestAttribute,categorical_predictors[bestAttribute],bestCutPoint,cutCategories)
        current_Node.setChildren(children)
        for child in current_Node.getChildren():
            DFSGrowTree(child,min_samples_split,categorical_predictors,evaluationFunction,features_left)
        return
    else:
        return    