import numpy as np
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class Layer:

    def __init__(self, model, curr_size, next_size, learning_rate=.1):
        self.model = model
        self.weights = np.random.rand(curr_size, next_size)
        self.learning_rate = learning_rate

    def forward(self, X):
        '''Given some input vector, dot product with weight values and return result'''
        self.incoming = X
        act = X.dot(self.weights)
        act = sigmoid(act)
        self.outputs = act
        return act

    def backward(self, next_err):
        """Given some derivative error from next layer,
        update current layer's weights and
         return derivative error for layer before current one"""
        delta = dsigmoid(self.outputs)*(next_err)
        self.weights += self.learning_rate* self.incoming.T.dot(delta)
        # print self.incoming.T.dot(delta)[0][0]
        return delta.dot(self.weights.T)
