import numpy as np
import pandas as pd
import time


class NeuralNet(object):
    """Deep Neural Network with square error metrics"""
    def __init__(self, layers, activation='tanh', output='tanh',
                 bias=True, initial=0.1):
        self.acti = self.Choose(activation)
        self.out = self.Choose(output)
        self.layer = layers
        self.L = len(layers)
        self.bias = bias
        self.init = initial
        self.weights = {}
        self.gradient = {}
        self.initialize_weights()

    def Choose(self, activation):
        """Choose the activation function"""
        choose = {
            'tanh': self.tanh,
            'sigmoid': self.sigmoid,
            'ReLU': self.ReLU,
            'linear': self.linear
        }

        return choose[activation]

    def initialize_weights(self):
        """Initialize the weights W and b"""
        for i in range(self.L - 1):
            bound = self.init
            size = (self.layer[i], self.layer[i + 1])
            self.weights[i] = np.random.uniform(-bound, bound, size)
            self.gradient[i] = np.zeros((self.layer[i], self.layer[i + 1]))

            # bias parameters
            if self.bias:
                w_b = np.random.uniform(-bound, bound, self.layer[i + 1])
                self.weights[-1-i] = w_b
                self.gradient[-1-i] = np.zeros(self.layer[i + 1])

    def get_param(self):
        """Print out all the parameters of the NNet"""
        for i in range(self.L - 1):
            print(self.weights[i])

            if self.bias:
                print(self.weights[-1-i])

    def train(self, X, Y, batch=1, rate=0.1, epochs=100):
        """Train NNet with backpropagation and mini-batch"""
        for j in range(epochs):
            Index = np.random.permutation(len(X))
            b = 0

            for ind in Index:
                self.Forward(X[ind])
                self.Backward(X[ind], Y[ind])
                b += 1

                if b == batch:
                    self.Update(rate, batch)
                    b = 0

    def Forward(self, x):
        """Forward propagation to get output for all layers"""
        self.tmp_S = {}
        self.tmp_A = {}

        for i in range(self.L - 1):
            if i == 0:
                # layer 0
                S = np.dot(x, self.weights[i])
            else:
                S = np.dot(self.tmp_A[i - 1], self.weights[i])

            if self.bias:
                S += self.weights[-1-i]

            self.tmp_S[i] = S
            self.tmp_A[i] = self.acti(S) if i < self.L - 2 else self.out(S)

        return self.tmp_A[self.L - 2]

    def Backward(self, x, y):
        """Calculate and accumulate the gradients of each parameters"""
        diff_y_output = (y - self.tmp_A[self.L - 2])
        second_term = self.out(self.tmp_S[self.L - 2], True)
        delta = -2 * diff_y_output * second_term
        self.gradient[self.L - 2] += np.outer(self.tmp_A[self.L - 3], delta)

        # gradient with respect to bias weight parameters
        if self.bias:
            self.gradient[1 - self.L] += delta

        for i in range(self.L - 3):
            first = np.dot(self.weights[self.L-2-i], delta)
            second = self.out(self.tmp_S[self.L - 3 - i], True)
            delta = first * second

            if self.bias:
                self.gradient[2 - self.L + i] += delta

            grad = np.outer(self.tmp_A[self.L-4-i], delta)
            self.gradient[self.L - 3 - i] += grad

        delta = np.dot(self.weights[1], delta) * self.out(self.tmp_S[0], True)

        if self.bias:
            self.gradient[-1] += delta

        self.gradient[0] += np.outer(x, delta)

    def Update(self, rate, batch):
        """Update weight parameters and reset gradients to 0"""
        for i in range(self.L - 1):
            self.weights[i] -= rate * self.gradient[i] / batch
            self.gradient[i] *= 0

            if self.bias:
                self.weights[-1-i] -= rate * self.gradient[-1-i] / batch
                self.gradient[-1-i] *= 0

    def predict(self, X_t):
        """Use current parameters to predict"""
        return np.array([self.Forward(x) for x in X_t])

    def tanh(self, X, grad=False):
        """tanh activation function and its gradient"""
        return (1 - np.tanh(X)**2) if grad else np.tanh(X)

    def sigmoid(self, X, grad=False):
        """sigmoid activation function and its gradient"""
        E = np.exp(-X)
        return E / (1 + E)**2 if grad else 1 / (1 + E)

    def ReLU(self, X, grad=False):
        """ReLU activation function and its gradient"""
        return np.where(X > 0, 1, 0) if grad else np.where(X > 0, X, 0)

    def linear(self, X, grad=False):
        """linear activation function and its gradient"""
        return X * 0 + 1 if grad else X


def run_NeuralNet(archi, num_model=50, base_dir='./'):
    # load data
    Data = pd.read_csv(base_dir + 'Data/hw4_nnet_train.dat',
                       sep=' ', header=None)
    Xt = Data[[0, 1]].values
    yt = Data[2].values

    Test = pd.read_csv(base_dir + 'Data/hw4_nnet_test.dat',
                       sep=' ', header=None)
    Xtest = Test[[0, 1]].values
    ytest = Test[2].values

    print('Data loaded. Start training 50 Neural Nets ...')
    print('\tNeural Net neurons: %s' % ' - '.join(map(str, archi)))

    start = time.clock()
    R = []
    for i in range(num_model):
        NN = NeuralNet(archi)
        NN.train(Xt, yt, epochs=2000, rate=0.01)
        Prediction = NN.predict(Xtest)
        R.append(np.sum((Prediction.T * ytest) < 0) / len(Prediction))
        print('\tDone training %d-th Neural Net.' % (i + 1))

    Eout = np.mean(R)
    print('Using %.2f seconds. Averaged Error Rate = %.4f %%' %
          (time.clock() - start, Eout * 100))


def main():
    archi = [2, 8, 3, 1]
    run_NeuralNet(archi)


if __name__ == '__main__':
    main()
