import time

import numpy as np
import pandas as pd
import theano as th
import theano.tensor as T


class Autoencoder(object):

    def __init__(self, layers, activ='tanh', update='sgd', lr=0.0003,
                 batch=1, memo=0.3):
        self.x = T.matrix()
        self.y_hat = T.matrix()
        self.layers = layers
        self.activ, self.update = self.choose(activ, update)
        self.batch = batch
        self.lr = lr
        self.memo = memo
        self.weights = []
        self.biases = []
        self.auxiliary = []
        self.a_n = [self.x]

    def choose(self, activ, update):
        """Choose the activation and update function"""
        Acti = dict({'tanh': self.tanh, 'sigmoid': self.sigmoid,
                    'ReLU': self.ReLU, 'linear': self.linear})
        Upda = dict({'sgd': self.sgd, 'NAG': self.NAG,
                    'RMSProp': self.RMSProp, 'momentum': self.momentum})
        return Acti[activ], Upda[update]

    def architecture(self, cons, code_layer):
        """Build up the architecture by theano"""
        for i in range(len(self.layers)-1):
            # Initialize shared variables
            init_w = cons*np.random.randn(self.layers[i], self.layers[i+1])
            self.weights.append(th.shared(init_w))
            init_bias = cons*np.random.randn(self.layers[i+1])
            self.biases.append(th.shared(init_bias))

            # Building architecture
            a_before = T.dot(self.a_n[i], self.weights[i]) + \
                self.biases[i].dimshuffle('x', 0)
            a_next = self.activ(a_before)
            self.a_n.append(a_next)

        # help the optimization
        for param in (self.weights+self.biases):
            self.auxiliary.append(th.shared(np.zeros(param.get_value().shape)))

        self.encode = th.function([self.x], self.a_n[code_layer])
        self.decode = th.function([self.a_n[code_layer]], self.a_n[-1])

        # Calculate the cost and gradients
        Cost = (T.sum((self.a_n[-1]-self.y_hat)**2))/self.batch
        params = self.weights + self.biases
        grads = T.grad(Cost, params, disconnected_inputs='ignore')

        # Update parameters
        update_query = self.update(params, grads, self.auxiliary)
        self.gradient_2 = th.function(inputs=[self.x, self.y_hat],
                                      updates=update_query, outputs=Cost)

    def fit(self, X, code_layer=1, epoch=10, print_every=1, cons=0.3,
            denoise=0.005):
        """fitting the data (unsupervised learning)"""
        self.architecture(cons, code_layer)
        start = time.clock()
        self.Cost_Record = []
        for j in range(epoch):
            costs = 0
            rounds = int(X.shape[0]/self.batch)
            X_permuted = X[np.random.permutation(X.shape[0])]
            noise = np.random.randn(X.shape[0], X.shape[1])
            X_permuted += noise*X_permuted.std(axis=0)*denoise

            for i in range(rounds):
                batch_X = X_permuted[i*self.batch:(i+1)*self.batch]
                costs += self.gradient_2(batch_X, batch_X)

            self.Cost_Record.append(costs/rounds)

            if j % print_every == 0:
                print("Epoch %d ; Cost: %f; %f seconds used." %
                      (j+1, self.Cost_Record[-1], (time.clock()-start)))

    def encode(self, X):
        return self.encode(X)

    def decode(self, X):
        return self.decode(X)

    # Optimization methods:
    def sgd(self, para, grad, _):
        """optimized by gradient descent"""
        return [(para[ix], para[ix] - self.lr*grad[ix])
                for ix in range(len(grad))]

    def NAG(self, para, grad, Real):
        """optimized by Nesterov accelerated gadient(NAG)"""
        updates = []
        for ix in range(len(grad)):
            # grad[ix] = T.clip(grad[ix],-1,1)
            gradient = -(self.lr/self.batch)*grad[ix]
            spy_position = (1+self.memo)*(para[ix]+gradient)-self.memo*Real[ix]
            updates.append((para[ix], spy_position))
            updates.append((Real[ix], para[ix]+gradient))
        return updates

    def momentum(self, para, grad, Momentum):
        """optimized by momentum"""
        updates = []
        for ix in range(len(grad)):
            # grad[ix] = T.clip(grad[ix],-1,1)
            grad_step = (self.lr / self.batch)*grad[ix]
            direction = (self.memo)*Momentum[ix] - grad_step
            updates.append((para[ix], para[ix]+direction))
            updates.append((Momentum[ix], direction))
        return updates

    def RMSProp(self, para, grad, Sigma_square):
        """optimized by RMSProp"""
        updates = []
        alpha = self.memo
        for ix in range(len(grad)):
            # grad[ix] = T.clip(grad[ix],-1,1)
            gradient = grad[ix]/self.batch
            Factor = Sigma_square[ix]*alpha+(1-alpha)*(gradient**2)
            direction = -(self.lr)*gradient/(T.sqrt(Factor)+0.001)
            updates.append((para[ix], para[ix]+direction))
            updates.append((Sigma_square[ix], Factor))
        return updates

    # Activation functions:
    def tanh(self, Z):
        return T.tanh(Z)

    def ReLU(self, Z):
        return T.switch(Z < 0, 0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + T.exp(-Z))

    def linear(self, Z):
        return Z


def run_autoencoder(archi, base_dir='./'):
    np.set_printoptions(4)
    data = pd.read_csv(base_dir + 'Data/train.dat',
                       header=None, delim_whitespace=True)

    Demo = [('tanh', 'RMSProp', 0.5), ('tanh', 'momentum', 0.5),
            ('tanh', 'NAG', 0.2), ('sigmoid', 'sgd', 0.5),
            ('ReLU', 'NAG', 0.2), ('sigmoid', 'RMSProp', 0.5)]

    for act, opti, mem in Demo:
        coder = Autoencoder(archi, batch=4,
                            activ=act, update=opti, memo=mem, lr=0.003)

        print(act, '/', opti, ':')
        coder.fit(data.values, code_layer=4, epoch=501, print_every=100)
        code = coder.encode(data.values)

        print('encode:\n', code[:3])
        print('decode:\n', coder.decode(code)[:3])
        print('original:\n', data.values[:3])


def main():
    archi = [9, 64, 32, 16, 2, 16, 32, 64, 9]
    run_autoencoder(archi)


if __name__ == "__main__":
    main()
