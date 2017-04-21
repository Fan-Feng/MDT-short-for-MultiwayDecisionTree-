from C4dot5 import DecisionTree

rows =  [[1,2,2,'T'],
        [1,2,7,'F'],
        [2,1,3,'T'],
         [2,1,5,'F']]

DTClassifier = DecisionTree()
DTClassifier.fit(rows)
print("Decision tree successfully fitted")