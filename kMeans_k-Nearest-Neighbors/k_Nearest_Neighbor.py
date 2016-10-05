import numpy as np
import pandas as pd
import time

class kNN(object):
    """Uniform voting by k nearest neighbors as predictions"""
    def __init__(self,k):
        self.k = k
        
    def train(self,X,y):
        self.Xtrain = X
        self.ytrain = y
        
    def predict(self,Xtest):
        """Find the k nearest neighbor of the input and do uniform voting"""
        prediction = []
        for xtest in Xtest:
            Ind_Dist = [(float("inf"),0)]*self.k  #initialize k nearest neighbors
            for ind,xtrain in enumerate(self.Xtrain):
                dist = np.sum((xtest-xtrain)**2)
                if dist <= Ind_Dist[-1][0]:
                    Ind_Dist[-1] = (dist,ind)
                    Ind_Dist.sort()
            predict = 1 if sum([self.ytrain[ind] for _,ind in Ind_Dist]) >= 0 else -1
            prediction.append(predict)
        return np.array(prediction)

Train = pd.read_csv('Data/hw4_knn_train.dat',sep=' ',header=None)
Test = pd.read_csv('Data/hw4_knn_test.dat',sep=' ',header=None)
print('Data loaded. Start predicting...')

start = time.clock()

K = 5
one_neighbor = kNN(K)
one_neighbor.train(Train[list(range(9))].values,Train[9].values)
print('Using k-nearest-neighbor with k = %d'%K)

Predict = one_neighbor.predict(Train[list(range(9))].values)
Ein = sum(Predict!=Train[9].values)/len(Predict)
print('Error rate on training set: %.2f %%'%(100*Ein))

Predict = one_neighbor.predict(Test[list(range(9))].values)
Eout = sum(Predict!=Test[9].values)/len(Predict)
print('Error rate on test set: %.2f %%'%(100*Eout))
print('Using %.2f seconds'%(time.clock()-start))