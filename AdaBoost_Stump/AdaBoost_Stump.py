# Training set:
# https://d396qusza40orc.cloudfront.net/ntumltwo/hw2_data/hw2_adaboost_train.dat
# Testing set:
# https://d396qusza40orc.cloudfront.net/ntumltwo/hw2_data/hw2_adaboost_test.dat

import numpy as np
import pandas as pd
import time


def memo(f):
    """Memoization decorator, Used to accelerate the retrieval"""
    cache = {}

    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        # Some elements of args unhashable
        except TypeError:
            return f(args)

    _f.cache = cache
    return _f


@memo
def stump(args):
    s, i, t, train_data = args
    """Decision stump for given direction s, dimension i, and threshold t"""
    return train_data.apply(lambda x: s * ((x[i] > t) * 2 - 1), axis=1)


def Accuracy(s, i, theta, w, train_data):
    """Calculate accuracy on training set for given decision stump"""
    index = stump(s, i, theta, train_data) == train_data[2]
    return (np.dot(np.array(index * 1), w), index)


def make_thresholds(L):
    """Given values of one dimension, let midpoints as thresholds"""
    LS = [min(L) - 1] + sorted(L)
    return [(LS[i] + LS[i + 1]) / 2 for i in range(len(LS) - 1)]


def AdaBoost_Training(train_data, T):
    """T is the number iterations, train an AdaBoost binary classifer"""
    # Initialize weight vector
    train_data['w'] = np.ones((100,)) / 100
    alpha = []
    g = []
    Thr = []

    # Compute threshold
    for i in range(2):
        Thr.append(make_thresholds(train_data[i]))

    for r in range(T):
        Max_Weighted_Accu = 0
        index = []
        w0 = train_data['w'].values

        for i in range(2):
            for t in Thr[i]:
                for s in [1, -1]:
                    A, ind = Accuracy(s, i, t, w0, train_data)

                    if A > Max_Weighted_Accu:
                        Max_Weighted_Accu, index = A, ind
                        best = s, i, t

        r_2 = Max_Weighted_Accu / (sum(w0) - Max_Weighted_Accu)
        Rescale_Factor = np.sqrt(r_2)

        # Rescaling the weight vector
        train_data['w'][index] /= Rescale_Factor
        train_data['w'][~index] *= Rescale_Factor

        alpha.append(np.log(Rescale_Factor))
        g.append(best)

    return g, alpha


def Predict_Accu_Train(g, alpha, T, train_data):
    G = np.zeros((len(train_data),))
    for i in range(T):
        G += np.array(stump(*g[i])) * alpha[i]

    return sum(((G > 0) * 2 - 1) == train_data[2]) / len(train_data)


@memo
def predict_stump(args):
    s, i, t, test_data = args
    return test_data.apply(lambda x: s * ((x[i] > t) * 2 - 1), axis=1)


def Predict_Accu_Test(g, alpha, T, test_data):
    G = np.zeros((len(test_data),))
    for i in range(T):
        params = list(g[i]) + [train_data]
        G += np.array(predict_stump(*params)) * alpha[i]

    return sum(((G > 0) * 2 - 1) == test_data[2]) / len(test_data)


def main():
    train_data = pd.read_csv('Data/hw2_adaboost_train.dat', sep=' ',
                             header=None)
    test_data = pd.read_csv('Data/hw2_adaboost_test.dat', sep=' ', header=None)

    print('Start Training...')
    start = time.clock()

    T = 300
    g, alpha = AdaBoost_Training(train_data, T)
    print('Done Training, %f seconds.' % (time.clock() - start))

    train_accu = Predict_Accu_Train(g, alpha, T, train_data)
    print('Accuracy on Training set: %.2f %%' % (100 * train_accu))

    test_accu = Predict_Accu_Test(g, alpha, T, test_data)
    print('Accuracy on Testing set: %.2f %%' % (100 * test_accu))

    min_err = min(list(map(lambda x: 1 / (np.exp(2 * x) + 1), alpha)))
    print('Smallest error of all stumps (train) %.4f %%' % (100 * min_err))

    one_accu = sum(predict_stump(*g[0]) == test_data[2]) / 10.0
    print('Accuracy on Testing set of one stump: %.2f %%' % (one_accu))


if __name__ == '__main__':
    main()
