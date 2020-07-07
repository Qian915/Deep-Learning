import numpy as np
from Layers.Base import BaseLayer
from Layers import Initializers
from Layers import Helpers
from Optimization import Optimizers
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights, self.bias = self.initialize(Initializers.Constant(1), Initializers.Constant(0))
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None
        self.input = None
        self.x_norm = None

    def initialize(self, weights_initializer, bias_initializer):
        # weights, bias: vector of channels length
        self.weights = weights_initializer.initialize((1, self.channels), 1, self.channels)
        self.bias = bias_initializer.initialize((1, self.channels,), 1, self.channels)
        return self.weights, self.bias

    def forward(self, input_tensor):
        # convolutional: 4D -> 2D
        original_tensor = copy.deepcopy(input_tensor)
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
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

        # convolutional: 2D -> 4D
        if len(original_tensor.shape) == 4:
            out = self.reformat(out)
        return out

    def backward(self, error_tensor):
        # convolutional: 4D -> 2D
        original_tensor = copy.deepcopy(error_tensor)
        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)

        # grad w.r.t. weights & bias
        self.gradient_weights = np.sum(error_tensor * self.x_norm, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        # 2D -> 4D
        if len(original_tensor.shape) == 4:
            self.gradient_weights = self.reformat(self.gradient_weights)
            self.gradient_bias = self.reformat(self.gradient_bias)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        # grad w.r.t. input
        error = Helpers.compute_bn_gradients(error_tensor, self.input, self.weights, self.mu, self.var, eps=np.finfo(float).eps)

        # convolutional: 2D -> 4D
        if len(original_tensor.shape) == 4:
            error = self.reformat(error)
        return error

    def reformat(self, tensor):
        # 4D -> 2D
        if len(tensor.shape) == 4:
            self.b = tensor.shape[0]
            self.h = tensor.shape[1]
            self.m = tensor.shape[2]
            self.n = tensor.shape[3]
            out = tensor.reshape(self.b, self.h, self.m*self.n)
            out = out.transpose(0, 2, 1)
            out = out.reshape(self.b*self.m*self.n, self.h)
        # 2D -> 4D
        else:
            out = tensor.swapaxes(0, 1)
            out = out.reshape(self.h, self.b, self.m*self.n)
            out = out.transpose(1, 0, 2)
            out = out.reshape(self.b, self.h, self.m, self.n)

        return out
