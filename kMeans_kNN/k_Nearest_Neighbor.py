import numpy as np
import pandas as pd
import time


class kNN(object):
    """Uniform voting by k nearest neighbors as predictions"""
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.Xtrain = X
        self.ytrain = y

    def predict(self, Xtest):
        """Find the k nearest neighbor of the input and do uniform voting"""
        prediction = []
        for xtest in Xtest:
            # initialize k nearest neighbors
            Ind_Dist = [(float("inf"), 0)] * self.k

            for ind, xtrain in enumerate(self.Xtrain):
                dist = np.sum((xtest - xtrain)**2)

                if dist <= Ind_Dist[-1][0]:
                    Ind_Dist[-1] = (dist, ind)
                    Ind_Dist.sort()

            voted_result = sum([self.ytrain[ind] for _, ind in Ind_Dist])
            predict = 1 if voted_result >= 0 else -1
            prediction.append(predict)

        return np.array(prediction)


def run_kNN(base_dir='./'):
    train_data = pd.read_csv(base_dir + 'Data/hw4_knn_train.dat',
                             sep=' ', header=None)
    test_data = pd.read_csv(base_dir + 'Data/hw4_knn_test.dat',
                            sep=' ', header=None)
    print('Data loaded. Start predicting...\n')

    start = time.clock()

    K = 5
    one_neighbor = kNN(K)
    one_neighbor.train(train_data[list(range(9))].values, train_data[9].values)
    print('Using k-nearest-neighbor with k = %d' % K)

    train_pred = one_neighbor.predict(train_data[list(range(9))].values)
    Ein = sum(train_pred != train_data[9].values) / len(train_pred)
    print('\tError rate on training set: %.2f %%' % (100 * Ein))

    test_pred = one_neighbor.predict(test_data[list(range(9))].values)
    Eout = sum(test_pred != test_data[9].values) / len(test_pred)
    print('\tError rate on test set: %.2f %%' % (100 * Eout))

    print('\nUsing %.2f seconds' % (time.clock() - start))


def main():
    run_kNN()


if __name__ == '__main__':
    main()
