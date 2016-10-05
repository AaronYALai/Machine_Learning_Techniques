import numpy as np
import pandas as pd
import time

from DecisionTree import DTree


# Random Forest inherits C&RT decision tree
class Random_Forest(DTree):
    def __init__(self, d=2):
        self.dim = d
        # Collection of Decision Trees
        self.Collect = {}

    def build_forest(self, df, num=100, max_depth=100):
        """Train decision trees on many bootstrapped datasets"""
        N = len(df)
        for i in range(num):
            Tree = DTree(self.dim)
            Tree.build_tree(self.Bootstrap(df, N), max_depth=max_depth)
            self.Collect[i] = Tree

    def Bootstrap(self, df, N):
        """Bootstrapping with the size the same as the original dataset"""
        return df.sample(N, replace=True)

    def predict(self, X):
        """Uniform voting to determine which class the input belongs to"""
        s = sum([self.Collect[i].predict(X) for i in range(len(self.Collect))])
        return 1 if s >= 0 else -1


def train_Forest(train_data, test_data, num, max_depth=100):
    RF = Random_Forest()
    RF.build_forest(train_data, num, max_depth)

    train_pred = [RF.predict(X) for X in train_data[[0, 1]].values]
    train_accu = sum(train_pred == train_data.y) * 100 / len(train_data)

    test_pred = [RF.predict(X) for X in test_data[[0, 1]].values]
    test_accu = sum(test_pred == test_data.y) * 100 / len(test_data)

    return train_accu, test_accu


def main():
    # Loading Data
    train_data = pd.read_csv('Data/hw3_train.dat', sep=' ',
                             header=None, names=[0, 1, 'y'])
    test_data = pd.read_csv('Data/hw3_test.dat', sep=' ',
                            header=None, names=[0, 1, 'y'])

    print("Train on 1 Random Forest with 300 trees.")
    Forest_Start = time.clock()
    train_accu, test_accu = train_Forest(train_data, test_data, 300)
    print("\tAccuracy on Train set: %.3f %%" % train_accu)
    print("\tAccuracy on Test set: %.3f %%" % test_accu)
    print("Using %.3f seconds\n" % (time.clock() - Forest_Start))

    print("Train 100 Forests to get averaged accuracy.")
    Forests_Start = time.clock()
    Train_Accu = []
    Test_Accu = []

    for i in range(100):
        train_accu, test_accu = train_Forest(train_data, test_data, 300)
        Train_Accu.append(train_accu)
        Test_Accu.append(test_accu)

    print("\tAccuracy on Train set: %.3f %%" % (np.mean(Train_Accu)))
    print("\tAccuracy on Test set: %.3f %%" % (np.mean(Test_Accu)))
    print("Using %.3f seconds\n" % (time.clock() - Forests_Start))

    print("Get averaged accuracy on 100 Forests")
    print(", whose trees have only 1 branch(Pruned).")
    Pruned_Forests_Start = time.clock()
    Train_Accu = []
    Test_Accu = []

    for i in range(100):
        train_accu, test_accu = train_Forest(train_data, test_data, 300, 1)
        Train_Accu.append(train_accu)
        Test_Accu.append(test_accu)

    print("\tAccuracy on Train set: %.3f %%" % (np.mean(Train_Accu)))
    print("\tAccuracy on Test set: %.3f %%" % (np.mean(Test_Accu)))
    print("Using %.3f seconds" % (time.clock() - Pruned_Forests_Start))


if __name__ == '__main__':
    main()
