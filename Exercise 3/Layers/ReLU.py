import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    # constructor
    def __init__(self):
        super().__init__()
        self.input = None

    # f(x) = max(0, x)
    def forward(self, input_tensor):
        self.input = input_tensor
        return np.where(input_tensor > 0, input_tensor, np.zeros_like(input_tensor))

    def backward(self, error_tensor):
        return np.where(self.input <= 0, np.zeros_like(error_tensor), error_tensor)

