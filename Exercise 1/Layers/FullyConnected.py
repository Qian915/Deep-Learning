import numpy as np
from Optimization import Optimizers


class FullyConnected:

    # constructor
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # private member: _optimizer
        self._optimizer = None

        self.grad_weights = None

        # w: (input_size+1, output_size)
        self.weights = np.random.uniform(0, 1, (self.input_size+1, self.output_size))

        # x:(batch_size, input_size+1)
        self.input = None

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
        # input (+bias)
        bias = np.ones((input_tensor.shape[0], 1))
        self.input = np.concatenate((input_tensor, bias), axis=1)

        out = np.dot(self.input, self.weights)     # can't concatenate ??????????????????????????????????
        return out

    def backward(self, error_tensor):
        # x_grad
        x_grad = np.dot(error_tensor, self.weights.T)

        # w_grad
        self.grad_weights = np.dot(self.input.T, error_tensor)

        # update weights
        self.optimizer = Optimizers.Sgd(1e-3)
        self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return x_grad, self.gradient_weights




