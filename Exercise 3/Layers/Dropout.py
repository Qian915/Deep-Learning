import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        # p: keep!
        self.probability = probability
        self.activations = None

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.activations = np.random.binomial(n=1, p=self.probability, size=input_tensor.shape) / self.probability
            out = input_tensor * self.activations
        # skip dropout layer in testing time
        else:
            return input_tensor
        return out

    def backward(self, error_tensor):
        if not self.testing_phase:
            error_tensor *= self.activations
        return error_tensor
