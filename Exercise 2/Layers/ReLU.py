import numpy as np

class ReLU:

    # constructor
    def __init__(self):
        self.input = None

    # f(x) = max(0, x)
    def forward(self, input_tensor):
        self.input = input_tensor
        return np.where(input_tensor > 0, input_tensor, np.zeros_like(input_tensor))

    def backward(self, error_tensor):
        return np.where(self.input <= 0, np.zeros_like(error_tensor), error_tensor)

