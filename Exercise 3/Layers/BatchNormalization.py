import numpy as np
from Layers.Base import BaseLayer
from Layers import Initializers
from Layers import Helpers
from Optimization import Optimizers


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights, self.bias = self.initialize(Initializers.Constant(1), Initializers.Constant(0))
        self.optimizer = None
        self.input = None
        self.x_norm = None

    def initialize(self, weights_initializer, bias_initializer):
        # weights, bias: vector of channels length
        self.weights = weights_initializer.initialize((1, self.channels), 1, self.channels)
        self.bias = bias_initializer.initialize((1, self.channels,), 1, self.channels)
        return self.weights, self.bias

    def forward(self, input_tensor):
        self.input = input_tensor
        # initialize testing mean, var
        testing_mu = np.mean(input_tensor, axis=0)
        testing_var = np.var(input_tensor, axis=0)
        if not self.testing_phase:
            self.mu = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.x_norm = (input_tensor - self.mu) / np.sqrt(self.var + np.finfo(float).eps)
            out = self.weights * self.x_norm + self.bias
            # store testing mean & var while training
            # ????? online estimation ????? forward(self, input_tensor, cache)?
            testing_mu = .8 * testing_mu + .2 * self.mu
            testing_var = .8 * testing_var + .2 * self.var
        else:
            self.x_norm = (input_tensor - testing_mu) / np.sqrt(testing_var + np.finfo(float).eps)
            out = self.weights * self.x_norm + self.bias
        return out

    def backward(self, error_tensor):
        # grad w.r.t. weights & bias
        grad_weights = np.sum(error_tensor * self.x_norm, axis=0)
        grad_bias = np.sum(error_tensor, axis=0)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, grad_weights)
            self.bias = self.optimizer.calculate_update(self.bias, grad_bias)

        # grad w.r.t. input
        error = Helpers.compute_bn_gradients(error_tensor, self.input, self.weights, self.mu, self.var, eps=np.finfo(float).eps)
        return error
