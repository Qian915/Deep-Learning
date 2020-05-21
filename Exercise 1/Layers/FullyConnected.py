import numpy as np
from Optimization import Optimizers


class FullyConnected:

    # constructor
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # protected member: _optimizer
        self._optimizer = Optimizers.Sgd(np.random.uniform(0, 1))

        self.grad_weights = None

        # w: (input_size+1, output_size)
        self.weights = np.random.uniform(0, 1, (self.input_size+1, self.output_size))

        # x:(batch_size, input_size+1)
        self.input = np.ndarray

    # property: optimizer, gradient_weights
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        return self.grad_weights

    def forward(self, input_tensor):
        input_tensor = np.array(input_tensor)    # input_tensor is of type tuple
        # input with bias
        bias = np.ones((np.shape(input_tensor)[0], 1))
        self.input = np.append(input_tensor, bias, axis=1)

        out = np.dot(self.input, self.weights)     # (50,4) (5,3) ??????????????
        return out

    def backward(self, error_tensor):
        error_tensor = np.array(error_tensor)     # error_tensor is of type tuple
        # x_grad w/o bias
        weights_no_bias = self.weights[0:-1, :]
        error = np.dot(error_tensor, weights_no_bias.T)

        # w_grad
        self.grad_weights = np.dot(self.input.T, error_tensor)

        # update weights
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error
