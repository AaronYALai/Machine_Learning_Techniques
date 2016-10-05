import time

import numpy as np
import pandas as pd


# Gradient Boosting Decision Tree(C&RT) with Gini Impurity


class GBDTree(object):

    def __init__(self):
        self.trees = {}       # Record each decision tree as base learners
        self.Branch = {}      # Record where and how to branch
        self.Value = {}       # Record the predict values at leaves

    def fit(self, X, y, T=50, depth=100):
        """Gradient boosting machine on decision tree for regression"""
        self.dim = X.shape[1]
        self.s = np.zeros((X.shape[0], 1))
        X_cols = list(range(self.dim))
        y = y.reshape(X.shape[0], 1)

        for i in range(T):
            data = np.hstack((X, y-self.s))
            df = pd.DataFrame(data=data, columns=X_cols+['y'])
            self.build_tree(df, depth=depth)
            g_t = np.array([self.tree_predict(x) for x in df[X_cols].values])

            if np.dot(g_t, g_t) != 0:
                alpha = np.dot(g_t, df.y.values.ravel()) / np.dot(g_t, g_t)
            else:
                alpha = 0

            self.trees[i] = (alpha, self.Branch, self.Value)
            self.clean()
            self.s += alpha*(g_t.reshape(X.shape[0], 1))

        print('Done.')

    def clean(self):
        self.Branch = {}
        self.Value = {}

    def predict(self, X):
        output = np.zeros(X.shape[0])
        for alpha, branch, value in self.trees.values():
            self.Branch = branch
            self.Value = value
            output += alpha*np.array([self.tree_predict(x) for x in X])

        self.clean()
        return output

    def build_tree(self, df, layer=0, side=0, depth=100):
        """Build the decision tree by recursively branching by Gini impurity"""
        # Cannot be branched anymore
        if len(set(df.y)) == 1:
            self.Value[(layer, side)] = df.y.values[0]

        # Depth reaches the limit
        elif layer >= depth:
            self.Value[(layer, side)] = 2 * (sum(df.y.values) >= 0) - 1

        else:
            best_d, best_val = self.branching(df, layer, side)
            # Left hand side
            self.build_tree(df[df[best_d] >= best_val],
                            layer + 1, 2 * side, depth)
            # Right hand side
            self.build_tree(df[df[best_d] < best_val],
                            layer + 1, 2 * side + 1, depth)

    def branching(self, df, layer, side):
        """find the value of i-th feature for the best branching"""
        min_err = 1
        for i in range(self.dim):
            ddf = df.sort_values(i)
            Y = ddf.y.values

            for j in range(1, len(ddf)):
                err = self.impurity(Y, j)

                if err <= min_err:
                    best_d, best_val, min_err = i, ddf.iloc[j][i], err

        # Record the best branching parameters at this node
        self.Branch[(layer, side)] = best_d, best_val

        return best_d, best_val

    def impurity(self, Y, j):
        """Gini impurity for binary classification"""
        # Neglect repeated entries
        if Y[j] == Y[j-1]:
            return 1

        Y1 = sum(Y[:j])
        Y2 = sum(Y[j:])
        N = len(Y)
        T1 = j**2 - (Y1)**2
        T2 = (N-j)**2 - (Y2)**2

        return (T1 / j + T2 / (N-j)) / N

    def tree_predict(self, X_t, layer=0, side=0):
        """Predict the class X_t belongs to"""
        if (layer, side) in self.Value:
            return self.Value[(layer, side)]
        else:
            branch_d, branch_val = self.Branch[(layer, side)]
            C = 0 if X_t[branch_d] >= branch_val else 1
            return self.tree_predict(X_t, layer + 1, 2*side + C)


def run_GBDT(base_dir='./'):
    Train = pd.read_csv(base_dir + 'Data/hw3_train.dat', sep=' ', header=None,
                        names=[0, 1, 'y'])
    X_train = Train[[0, 1]].values
    y_train = np.ravel(Train[['y']])

    Test = pd.read_csv(base_dir + 'Data/hw3_test.dat', sep=' ', header=None,
                       names=[0, 1, 'y'])
    X_test = Test[[0, 1]].values
    y_test = np.ravel(Test[['y']])

    Start = time.clock()

    GBDT = GBDTree()
    GBDT.fit(X_train, y_train, T=10, depth=3)

    y_predict = np.array([1 if p >= 0 else -1 for p in GBDT.predict(X_train)])
    print("\tAccuracy on Train set: %.3f %%" %
          (sum(y_predict == y_train)*100 / Train.shape[0]))

    ytest_predict = [1 if p >= 0 else -1 for p in GBDT.predict(X_test)]
    print("\tAccuracy on Test set: %.3f %%" %
          (sum(np.array(ytest_predict) == y_test)*100 / Test.shape[0]))

    print("\nUsing %.3f seconds" % (time.clock()-Start))


def main():
    run_GBDT()


if __name__ == '__main__':
    main()
