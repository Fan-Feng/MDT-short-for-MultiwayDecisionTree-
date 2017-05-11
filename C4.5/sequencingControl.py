# coding: utf-8

# In[72]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import time,datetime
from sklearn import tree
import seaborn as sns


# user-defined module
#import dataAnalysis as da

def chillerCombinationExtract(data,c = 0.05):
    # c: on-off threshold
    # That is when electricity consumption is less than c*MaxPower,
    # this chiller can be viewed as closed
    maX = data.max(axis = 0)
    idx = data > c * maX
    result = bi2de(idx)
    return result
def chillerCombinationExtract_2(data,cap,c = 0.05):
    # c: on-off threshold
    # That is when electricity consumption is less than c*MaxPower,
    # this chiller can be viewed as closed
    maX = data.max(axis = 0)
    idx = data > c * maX
    return np.dot(idx,cap)


# In[84]:

def plot_heatMap(y,outFN,lg = pd.DataFrame([])):
    idx = np.reshape(np.reshape(np.repeat(np.arange(0,96),366,axis = 0),(96,-1)),(-1,1),order = 'F').flatten().tolist()
    #Prepare data
    data_Status2 = pd.DataFrame([list(map(lambda ts:ts.month,y.iloc[:,0])),
                             list(map(lambda ts:ts.day,y.iloc[:,0])),
                             #list(map(lambda ts:ts.time(),data_Status.index)),
                             idx,
                             list(y.iloc[:,1])],dtype = float).transpose()
    #pivot table: 数据透视表
    data_Status2_lg4 = data_Status2.copy()
    if lg.shape[0]:
        data_Status3_lg4 = data_Status2_lg4.ix[lg,:].pivot_table(values = [3],index = [0,1],
                    columns = [2] )
    else:
        data_Status3_lg4 = data_Status2_lg4.pivot_table(values = [3],index = [0,1],
                    columns = [2] )
        
    #plot
    fig = plt.figure(figsize = (10,10),dpi = 600)
    #sns.heatmap(np.array(data_Status3))
    ax = sns.heatmap(np.array(data_Status3_lg4), cmap = 'terrain_r',cbar=True,annot = False)
    ax.set_xticks([0,24,48,72,96])
    ax.set_xticklabels(['00:00','06:00',"12:00","18:00","23:00"], rotation = 'horizontal' )
    ax.set_yticks([31,59,90,120,151,181,212,243,273,304,334,365])
    #ax.set_yticks([30,60])
    ax.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Noc','Dec'][::-1], 
                       rotation = 'horizontal' ,va = "center")
    ax.set_ylabel("Date")
    ax.set_xlabel("time")
    plt.savefig(outFN,dpi = 600)


# In[81]:

def bi2de(DaIn):
    # This function convert boolean DataFrame to decimal number array
    
    ##This original implementation is sort of clumsy
    #DaOut = list(map(lambda i:int(''.join(list(map(lambda b: str(int(b)),DaIn.iloc[i,:]))), base = 2) ,range(DaIn.shape[0])))
    multiplier = [2**i for i in range(DaIn.shape[1])][::-1]
    return DaIn.dot(multiplier)
