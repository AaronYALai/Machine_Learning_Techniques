#Training set: https://d396qusza40orc.cloudfront.net/ntumltwo/hw2_data/hw2_adaboost_train.dat
#Testing set: https://d396qusza40orc.cloudfront.net/ntumltwo/hw2_data/hw2_adaboost_test.dat

import numpy as np
import pandas as pd
import time

Train = pd.read_csv('hw2_adaboost_train.dat',sep=' ',header=None)

def memo(f): 
    """Memoization decorator, Used to accelerate the retrieval"""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError: #Some elements of args unhashable
            return f(args)
    _f.cache = cache
    return _f

@memo
def stump(s,i,t):
    """Decision stump for given direction s, dimension i, and threshold t"""
    return Train.apply(lambda x: s*((x[i] > t)*2-1),axis=1)

def Accuracy(s,i,theta,w):
    """Calculate accuracy on training set for given decision stump"""
    index = stump(s,i,theta)==Train[2]
    return (np.dot(np.array(index*1),w),index)

def make_thresholds(L):
    """Given values of one dimension, let midpoints as thresholds"""
    LS = [min(L)-1]+sorted(L)
    return [(LS[i]+LS[i+1])/2 for i in range(len(LS)-1)]

def AdaBoost_Training(Train,T):
    """Given training set as a pandas dataframe and the iterations, train an AdaBoost binary classifer"""
    #Initialize weight vector
    Train['w'] = np.ones((100,))/100  
    alpha = []; g = []; Thr = []

    #Compute threshold
    for i in range(2):
        Thr.append(make_thresholds(Train[i])) 

    for r in range(T):
        Max_Weighted_Accu = 0; index = []; w0 = Train['w'].values
        for i in range(2):
            for t in Thr[i]:
                for s in [1,-1]:
                    A,ind = Accuracy(s,i,t,w0)

                    if A > Max_Weighted_Accu:
                        Max_Weighted_Accu, index = A, ind
                        best = s, i, t

        Rescale_Factor = np.sqrt(Max_Weighted_Accu/(sum(w0)-Max_Weighted_Accu))
        Train['w'][index] /= Rescale_Factor   #Rescaling the weight vector
        Train['w'][~index] *= Rescale_Factor
        alpha.append(np.log(Rescale_Factor))
        g.append(best)
        
    return g,alpha

def Predict_Accu_Train(g,alpha,T):
    G = np.zeros((len(Train),))
    for i in range(T):
        G += np.array(stump(*g[i]))*alpha[i]
    return sum(((G>0)*2-1)==Train[2])/len(Train)

@memo
def predict_stump(s,i,t):
    return Test.apply(lambda x: s*((x[i] > t)*2-1),axis=1)

def Predict_Accu_Test(g,alpha,T):
    G = np.zeros((len(Test),))
    for i in range(T):
        G += np.array(predict_stump(*g[i]))*alpha[i]
    return sum(((G>0)*2-1)==Test[2])/len(Test)

print('Start Training...')
start = time.clock()
T = 300
g,alpha = AdaBoost_Training(Train,T)
print('Done Training, %f seconds.'%(time.clock()-start))

Test = pd.read_csv('hw2_adaboost_test.dat',sep=' ',header=None)
print('Accuracy on Training set: %.2f %%'%100*Predict_Accu_Train(g,alpha,T))
print('Accuracy on Testing set: %.2f %%'%(100*Predict_Accu_Test(g,alpha,T)))

print('Smallest error rate of all stumps on training set %.4f %%' % (100*min(list(map(lambda x:1/(np.exp(2*x)+1),alpha)))))
print('Accuracy on Testing set of one stump: %.2f %%'%(sum(predict_stump(*g[0])==Test[2])/10))

