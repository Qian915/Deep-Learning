import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):

    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        out = 1 / (1 + np.exp(-input_tensor))
        self.activations = out
        return out

    def backward(self, error_tensor):
        out = self.activations * (1 - self.activations)
        return out * error_tensor



