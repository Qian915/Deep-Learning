import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.input = None

    def forward(self, input_tensor):
        self.input = input_tensor

    def backward(self, error_tensor):
        pass
