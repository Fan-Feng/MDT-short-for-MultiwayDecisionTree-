# MDT
MDT, short for **M**ultiway**D**ecision**T**ree

Multiway is derived from the classical decision tree algorithm--C4.5, while it adopts 
a multiway splitting method for numerical variables.That is, when finding a best splitting variable,
MDT fisrt applys a local entropy-based discretization method for each numerical variables.
Regarding each variable as categorical, and the multiway splitting method used in C4.5 is appliable now. 

This algorithm is still being developed. 

##usage

python MDT.py -d dataset.csv -o output.txt -c 0 0 0 1 0 0 -g 0.4


------------argument list---------

**-d,--dataset**: the name of dataset

**-o,-output   **: the name of output file

**-c           **: binary array, categoical_predictors. For instance, 0 0 0 1 0 0  means the 4th variable is categorical

**-g        **: min_gain, used in pruning tree 
                
