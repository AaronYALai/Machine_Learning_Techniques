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
    s, i, t, X = args
    """Decision stump for given direction s, dimension i, and threshold t"""
    return np.apply_along_axis(lambda x: s * ((x[i] > t) * 2 - 1), 1, X)


def accuracy(s, i, theta, w, X, y):
    """Calculate accuracy on training set for given decision stump"""
    index = (stump(s, i, theta, X) == y)
    return (np.dot(index * 1, w), index)


def make_thresholds(L):
    """Given values of one dimension, let midpoints as thresholds"""
    LS = [min(L) - 1] + sorted(L)
    return [(LS[i] + LS[i + 1]) / 2 for i in range(len(LS) - 1)]


def AdaBoost_Training(X, y, T):
    """T is the number iterations, train an AdaBoost binary classifer"""
    # Initialize weight vector
    weights = np.ones((X.shape[0],)) / X.shape[0]
    alpha = []
    g = []
    Thr = []

    # Compute threshold
    for i in range(2):
        Thr.append(make_thresholds(X[:, i]))

    for r in range(T):
        Max_Weighted_Accu = 0
        index = []

        for i in range(2):
            for t in Thr[i]:
                for s in [1, -1]:
                    A, ind = accuracy(s, i, t, weights, X, y)

                    if A > Max_Weighted_Accu:
                        Max_Weighted_Accu, index = A, ind
                        best = s, i, t

        r_2 = Max_Weighted_Accu / (sum(weights) - Max_Weighted_Accu)
        Rescale_Factor = np.sqrt(r_2)

        # Rescaling the weight vector
        weights[index] /= Rescale_Factor
        weights[~index] *= Rescale_Factor

        alpha.append(np.log(Rescale_Factor))
        g.append(best)

        if r % 10 == 9:
            print('\tNow is the %d-th iteration.' % (r + 1))

    return g, alpha, weights


def model_accuracy(g, alpha, T, X, y):
    G = np.zeros((X.shape[0],))
    for i in range(T):
        params = list(g[i]) + [X]
        G += np.array(stump(*params)) * alpha[i]

    return sum(((G > 0) * 2 - 1) == y) / X.shape[0]


def run(T):
    train_data = pd.read_csv('Data/hw2_adaboost_train.dat', sep=' ',
                             header=None)
    test_data = pd.read_csv('Data/hw2_adaboost_test.dat', sep=' ', header=None)

    X = train_data[train_data.columns[:-1]].values
    y = train_data[train_data.columns[-1]].values

    X_test = test_data[test_data.columns[:-1]].values
    y_test = test_data[test_data.columns[-1]].values

    print('Start Training...\n')
    start = time.clock()

    g, alpha, weights = AdaBoost_Training(X, y, T)

    train_accu = model_accuracy(g, alpha, T, X, y)
    print('\n\tAccuracy on Training set: %.2f %%' % (100 * train_accu))

    test_accu = model_accuracy(g, alpha, T, X_test, y_test)
    print('\tAccuracy on Testing set: %.2f %%' % (100 * test_accu))

    min_err = min(list(map(lambda x: 1 / (np.exp(2 * x) + 1), alpha)))
    print('\tSmallest error of all stumps (train) %.2f %%' % (100 * min_err))

    params = list(g[0]) + [X_test]
    one_accu = sum(stump(*params) == y_test) * 100 / len(y_test)
    print('\tAccuracy on Testing set of one stump: %.2f %%' % (one_accu))

    print('\nDone. Using %f seconds.' % (time.clock() - start))

if __name__ == '__main__':
    run(300)
