from unittest import TestCase

from AdaBoost_Stump.AdaBoost_Stump import run_AdaBoost
from Autoencoder.autoencoder import run_autoencoder
from DecisionTree_RandomForest.DecisionTree import run_DTree
from DecisionTree_RandomForest.RandomForest import run_RF
from Gradient_boosting_decision_tree.GBDT import run_GBDT
from kMeans_kNN.k_Nearest_Neighbor import run_kNN
from kMeans_kNN.KMeans import run_KMeans
from NeuralNet.NeuralNet import run_NeuralNet
from Radial_Basis_Function_Network.RBF_network import run_RBF
from Support_Vector_Regression.SVR_LSSVM import run_SVR


class Test_running(TestCase):

    def test_AdaBoost_Stump(self):
        run_AdaBoost(10, './AdaBoost_Stump/')

    def test_autoencoder(self):
        run_autoencoder([9, 16, 2, 16, 9], './Autoencoder/')

    def test_DTree(self):
        run_DTree('./DecisionTree_RandomForest/')

    def test_RandomForest(self):
        run_RF(n_tree=10, n_forest=10, base_dir='./DecisionTree_RandomForest/')

    def test_GBDT(self):
        run_GBDT('./Gradient_boosting_decision_tree/')

    def test_kNN(self):
        run_kNN('./kMeans_kNN/')

    def test_KMeans(self):
        run_KMeans('./kMeans_kNN/')

    def test_NeuralNet(self):
        run_NeuralNet([2, 3, 1], 5, './NeuralNet/')

    def test_RBF(self):
        run_RBF(100)

    def test_SVR(self):
        run_SVR('./Support_Vector_Regression/')
