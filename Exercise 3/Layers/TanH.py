import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):

    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        out = np.tanh(input_tensor)
        self.activations = out
        return out

    def backward(self, error_tensor):
        out = 1 - self.activations ** 2
        return out * error_tensor

