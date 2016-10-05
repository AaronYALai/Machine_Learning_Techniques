import pandas as pd
import time


# Decision Tree(C&RT) with Gini Impurity
class DTree(object):
    def __init__(self, d=2):
        # feature dimension of the input
        self.dim = d
        # Record where and how to branch
        self.Branch = {}
        # Record the predict values at leaves
        self.Value = {}

    def build_tree(self, df, layer=0, side=0, max_depth=100):
        """Build the decision tree by recursively branching by Gini impurity"""
        # Cannot be branched anymore
        if len(set(df.y)) == 1:
            self.Value[(layer, side)] = df.y.values[0]
        # Depth reaches the limit
        elif layer >= max_depth:
            self.Value[(layer, side)] = 2 * (sum(df.y.values) >= 0) - 1
        else:
            best_d, best_val = self.branching(df, layer, side)
            # Left hand side
            p = (df[df[best_d] >= best_val], layer + 1, 2 * side, max_depth)
            self.build_tree(*p)
            # Right hand side
            p = (df[df[best_d] < best_val], layer + 1, 2 * side + 1, max_depth)
            self.build_tree(*p)

    def branching(self, df, layer, side):
        """find the value of i-th feature for the best branching"""
        min_err = 1
        # Search for the best cut
        for i in range(self.dim):
            ddf = df.sort_values(i)
            Y = ddf.y.values

            for j in range(1, len(ddf)):
                err = self.impurity(Y, j)
                if err < min_err:
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
        T2 = (N - j)**2 - (Y2)**2

        return (T1 / j + T2 / (N - j)) / N

    def predict(self, X, layer=0, side=0):
        """recursively traversing: Predict which class the input belongs to"""
        if (layer, side) in self.Value:
            return self.Value[(layer, side)]
        else:
            branch_d, branch_val = self.Branch[(layer, side)]
            C = 0 if X[branch_d] >= branch_val else 1
            return self.predict(X, layer + 1, 2 * side + C)


def main():
    # Loading Data
    train_data = pd.read_csv('Data/hw3_train.dat', sep=' ',
                             header=None, names=[0, 1, 'y'])
    test_data = pd.read_csv('Data/hw3_test.dat', sep=' ',
                            header=None, names=[0, 1, 'y'])

    # Train on 1 decision tree
    Tree = DTree()
    Tree.build_tree(train_data)
    Tree_Start = time.clock()

    print("Train on 1 fully-grown Decision Tree.\nBranches:")
    for k in sorted(Tree.Branch.items()):
        print('\t', k)

    train_pred = [Tree.predict(X) for X in train_data[[0, 1]].values]
    train_accu = sum(train_pred == train_data.y) * 100 / len(train_data)
    print("\tAccuracy on Train set: %.3f %%" % train_accu)

    test_pred = [Tree.predict(X) for X in test_data[[0, 1]].values]
    test_accu = sum(test_pred == test_data.y) * 100 / len(test_data)
    print("\tAccuracy on Test set: %.3f %%" % test_accu)

    print("Using %.3f seconds" % (time.clock() - Tree_Start))


if __name__ == '__main__':
    main()
