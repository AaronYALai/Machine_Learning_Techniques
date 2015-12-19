import numpy as np
import pandas as pd
import time

#Loading Data
Train = pd.read_csv('hw3_train.dat',sep=' ',header=None,names=[0,1,'y'])
Test = pd.read_csv('hw3_test.dat',sep=' ',header=None,names=[0,1,'y'])

#Decision Tree(C&RT) with Gini Impurity
class DTree(object):
    def __init__(self, d=2): 
        self.dim = d       #feature dimension of the input
        self.Branch = {}   #Record where and how to branch
        self.Value = {}    #Record the predict values at leaves

    def build_tree(self,df,layer=0,side=0,depth=100): #Can use depth to prune the tree
        """Build the decision tree by recursively branching by Gini impurity""" 
        if len(set(df.y))==1:   #Cannot be branched anymore
            self.Value[(layer,side)] = df.y.values[0]
        elif layer >= depth:    #Depth reaches the limit
            self.Value[(layer,side)] = 2*(sum(df.y.values)>=0)-1
        else:
            best_d, best_val = self.branching(df,layer,side)
            self.build_tree(df[df[best_d]>=best_val],layer+1,2*side,depth)  #Left hand side
            self.build_tree(df[df[best_d]<best_val],layer+1,2*side+1,depth) #Right hand side

    def branching(self,df,layer,side):
        """find the value of i-th feature for the best branching"""     
        min_err = 1  
        for i in range(self.dim):
            ddf = df.sort_values(i)
            Y = ddf.y.values
            for j in range(1,len(ddf)): 
                err = self.impurity(Y,j)
                if err < min_err:
                    best_d, best_val, min_err = i, ddf.iloc[j][i], err
        self.Branch[(layer,side)] = best_d, best_val  #Record the best branching parameters at this node
        return best_d, best_val
    
    def impurity(self,Y,j):
        """Gini impurity for binary classification"""
        if Y[j] == Y[j-1]: #Neglect repeated entries
            return 1
        Y1 = sum(Y[:j]); Y2 = sum(Y[j:]); N = len(Y)
        T1 = j**2 - (Y1)**2
        T2 = (N-j)**2 - (Y2)**2
        return (T1/j + T2/(N-j))/N
    
    def predict(self,X,layer=0,side=0):
        """Predict which class the input belongs to by recursively traverse down the tree"""
        if (layer,side) in self.Value:
            return self.Value[(layer,side)]
        else:
            branch_d, branch_val = self.Branch[(layer,side)]
            C = 0 if X[branch_d] >= branch_val else 1
            return self.predict(X,layer+1,2*side+C)

#Train on 1 decision tree
Tree = DTree()
Tree.build_tree(Train)
Tree_Start = time.clock()
print("Train on 1 fully-grown Decision Tree.\nBranches:")
for k in sorted(Tree.Branch.items()):
    print(k)
print("Accuracy on Train set: %.3f %%"%(sum([Tree.predict(X) for X in Train[[0,1]].values]==Train.y)*100/len(Train)))
print("Accuracy on Test set: %.3f %%"%(sum([Tree.predict(X) for X in Test[[0,1]].values]==Test.y)*100/len(Test)))
print("Using %.3f seconds"%(time.clock()-Tree_Start))


#Random Forest inherits C&RT decision tree
class Random_Forest(DTree):
    def __init__(self, d=2):
        self.dim = d
        self.Collect = {}  #Collection of Decision Trees
    
    def build_forest(self,df,num=100,depth=100):
        """Train decision trees on many bootstrapped datasets"""
        N = len(df)
        for i in range(num):
            Tree = DTree(self.dim)
            Tree.build_tree(self.Bootstrap(df,N),depth=depth)
            self.Collect[i] = Tree
    
    def Bootstrap(self,df,N):
        """Bootstrapping with the size the same as the original dataset"""
        return df.sample(N,replace=True)
    
    def predict(self,X):
        """Uniform voting to determine which class the input belongs to"""
        s = sum([self.Collect[i].predict(X) for i in range(len(self.Collect))])
        return 1 if s >= 0 else -1


print("Train on 1 Random Forest with 300 trees.")
Forest_Start = time.clock()
RF = Random_Forest()
RF.build_forest(Train,300)
print("Accuracy on Train set: %.3f %%"%(sum([RF.predict(X) for X in Train[[0,1]].values]==Train.y)*100/len(Train)))
print("Accuracy on Test set: %.3f %%"%(sum([RF.predict(X) for X in Test[[0,1]].values]==Test.y)*100/len(Test)))
print("Using %.3f seconds"%(time.clock()-Forest_Start))


print("Train 100 Forests to get averaged accuracy.")
Forests_Start = time.clock()
Train_Accu = []; Test_Accu = []
N_Train = len(Train); N_Test = len(Test)

for i in range(100):
    RF = Random_Forest()
    RF.build_forest(Train,300)
    Train_Accu.append(sum([RF.predict(X) for X in Train[[0,1]].values]==Train.y)*100/N_Train)
    Test_Accu.append(sum([RF.predict(X) for X in Test[[0,1]].values]==Test.y)*100/N_Test)
    
print("Accuracy on Train set: %.3f %%"%(np.mean(Train_Accu)))
print("Accuracy on Test set: %.3f %%"%(np.mean(Test_Accu)))
print("Using %.3f seconds"%(time.clock()-Forests_Start))


print("Get averaged accuracy on 100 Forests whose trees have only 1 branch(Pruned).")
Pruned_Forests_Start = time.clock()
Train_Accu = []; Test_Accu = []
N_Train = len(Train); N_Test = len(Test)

for i in range(100):
    RF = Random_Forest()
    RF.build_forest(Train,300,1)
    Train_Accu.append(sum([RF.predict(X) for X in Train[[0,1]].values]==Train.y)*100/N_Train)
    Test_Accu.append(sum([RF.predict(X) for X in Test[[0,1]].values]==Test.y)*100/N_Test)
    
print("Accuracy on Train set: %.3f %%"%(np.mean(Train_Accu)))
print("Accuracy on Test set: %.3f %%"%(np.mean(Test_Accu)))
print("Using %.3f seconds"%(time.clock()-Pruned_Forests_Start))