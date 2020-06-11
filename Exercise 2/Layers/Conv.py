import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # padding: height & width
        self.phUp = np.ceil((self.convolution_shape.shape[1] - 1) / 2)
        self.phBottom = np.floor((self.convolution_shape.shape[1] - 1) / 2)
        self.pwLeft = np.ceil((self.convolution_shape.shape[2] - 1) / 2)
        self.pwRight = np.floor((self.convolution_shape.shape[2] - 1) / 2)

        self.input = None
        self.error = None

        self.weights = None
        self.bias = None

        self.grad_weights = None
        self.grad_bias = None

        self.weights_optimizer = None
        self.bias_optimizer = None

    @property
    def gradient_weights(self):
        return self.grad_weights

    @property
    def gradient_bias(self):
        return self.grad_bias

    @property
    def optimizer(self):
        return self.optimizer

    def initialize(self, weights_initializer, bias_initializer):
        # input_size, output_size ?????????
        w = weights_initializer.initialize((self.weights.shape[0]-1, self.weights.shape[1]), self.input_size, self.output_size)
        b = bias_initializer.initialize((1, self.weights.shape[1]), self.input_size, self.output_size)
        # concatenate w and b => weights with bias
        self.weights = np.vstack(w, b)
        return self.weights

    def forward(self, input_tensor):
        self.input = input_tensor

    def backward(self, error_tensor):
        self.error = error_tensor



