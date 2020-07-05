import numpy as np
from Layers.Base import BaseLayer
from Layers import Initializers


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights, self.bias = self.initialize(Initializers.Constant(1), Initializers.Constant(0))

 def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)

        return self.weights, self.bias
