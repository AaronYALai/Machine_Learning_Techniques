# Data:
# https://d396qusza40orc.cloudfront.net/ntumltwo/hw2_data/hw2_lssvm_all.dat
# first 400 rows as training set and last 100 rows as testing set

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


def kernel(x1, x2, gamma):
    """Guassian-RBF kernel"""
    return np.exp(-gamma * sum((x1 - x2)**2))


def g(beta, gamma, x):
    """One hypothesis"""
    val = sum([beta[i] * kernel(X[i], x, gamma) for i in range(len(beta))])
    return 1 if val >= 0 else -1


@memo
def kernel_matrix(N, gamma):
    """Compute the kernel matrix K(xi,xj)"""
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            K[i][j] = kernel(X[i], X[j], gamma)
            K[j][i] = K[i][j]

    return K


def beta_star(lamb_identity, K, train_data):
    """Calculate the optimal beta_star"""
    return np.dot(np.linalg.inv(lamb_identity + K), train_data[10].values)


def predicts(beta, gamma, data):
    """Predict by beta, gamma, and Data"""
    return np.array([g(beta, gamma, x) for x in data])


def main():
    LS_Data = pd.read_csv('Data/hw2_lssvm_all.dat', sep=' ',
                          header=None, skipinitialspace=True)

    train_data = LS_Data[:400]
    test_data = LS_Data[400:]

    print('Start Training...')
    start = time.clock()

    global X
    X = np.array(train_data[list(range(10))].values)
    X_test = np.array(test_data[list(range(10))].values)
    N = len(X)
    N_t = len(X_test)
    Max_in = 0
    Max_out = 0

    for lamb in [0.001, 1, 1000]:
        for gamma in [32, 2, 0.125]:
            lamb_identity = np.identity(N) * lamb
            K = kernel_matrix(N, gamma)
            beta = beta_star(lamb_identity, K, train_data)

            predict_in = predicts(beta, gamma, X)
            predict_out = predicts(beta, gamma, X_test)

            Accu_in = np.sum(np.array(predict_in) == train_data[10].values) / N
            correct_num = np.sum(np.array(predict_out) == test_data[10].values)
            Accu_out = correct_num / N_t

            if Accu_in > Max_in:
                best_in = lamb, gamma, beta
                Max_in = Accu_in

            if Accu_out > Max_out:
                best_out = lamb, gamma, beta
                Max_out = Accu_out

    print('\tBest Train Accuracy %.2f %%, with lambda %.3f and gamma %.3f.' %
          (100 * Max_in, best_in[0], best_in[1]))
    print('\tBest Test Accuracy %.2f %%, with lambda %.3f and gamma %.3f.' %
          (100 * Max_out, best_out[0], best_out[1]))
    print('Used %.2f seconds' % (time.clock() - start))


if __name__ == '__main__':
    main()
