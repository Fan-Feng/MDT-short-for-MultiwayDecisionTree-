# coding: utf-8

'''
    This .py file includes many data analysis tools that I created before.
    1) Outlier iteratively Eliminating(The original version is implemented
        in Matlab);
    2)
    
    author: Feng Fan
    email: fengfan6696@gmail.com
    date: 2017/2/21
'''
import re,time
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#get_ipython().magic('matplotlib inline')


def OutlierElimination(a,method = '3sigma'):
    '''
    % OutlierElimation eliminate outlier(inconsistent,etc.) from the data
    %  this function replace the outliers using interpolating 
    %  a : pandas data type:series
    %  m: eliminating method
    '''
    #a = pd.DataFrame(a)
    c = 5
    if method == '3sigma':
        (a, count) = threeSigma(a)
        num = 1
        while count>0:
            if num == 5:
                break
                
            try:
                (a, count) = threeSigma(a)
                num = num + 1
            except:
                print(a)
    elif method == '5point':
        (a, count) = fivePoint(a)
        num = 1
        while count>0:
            if num == 2:
                break
            (a, count) = fivePoint(a)
            num = num + 1
    
    return a
        
def threeSigma(a):
    ''' This function is a simple implementation of 3-sigma method
    '''
    mu = a.mean(skipna = True)
    sigma = a.std(skipna = True)
    UpEdge,LowEdge = mu + 3 * sigma, mu - 3 * sigma
    
    a1 = np.concatenate(([float('nan')],a[:-1]),axis = 0)
    a2 = np.concatenate((a[1:],[float('nan')]),axis = 0)
    A = pd.DataFrame([a1,a2]).transpose()
    
    logical_Idx = np.logical_or(a>UpEdge, a<LowEdge)
    count = sum(logical_Idx)
    if count:
        a.ix[logical_Idx] = A.ix[logical_Idx,:].mean(axis = 1,skipna = True)
    return a,count

def fivePoint(a):
    ''' This function is a simple implementation of 5-point method
    '''
    c = 5 # a factor to adjust the UpperEdge and LowerEdge
    a_Sorted = np.sort(a)
    l = len(a)
    percentile25Idx,percentile75Idx = round(l/4),round(3*l/4)
    
    median = round(l/2)
    UpEdge =a_Sorted[median] + c * (a_Sorted[percentile75Idx] - a_Sorted[percentile25Idx]);
    LowEdge = a_Sorted[median] - c * (a_Sorted[percentile75Idx] - a_Sorted[percentile25Idx]);

    a1 = np.concatenate(([float('nan')],a[:-1]),axis = 0)
    a2 = np.concatenate((a[1:],[float('nan')]),axis = 0)
    
    A = pd.DataFrame([a1,a2]).transpose()
    
    logical_Idx = np.logical_or(a>UpEdge, a<LowEdge)
    count = sum(logical_Idx)
    if count:
        a.loc[logical_Idx] = A.ix[logical_Idx,:].mean(axis = 1,skipna = True)
    return a,count




