
import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):

    def __init__self(self):
        super().__init__()
        self.input_shape = None
        self.batch_size = None

    def forward(self, input_tensor):
        self.input_shape = np.shape(input_tensor)[1:]
        self.batch_size = np.shape(input_tensor)[0]

        flatten = input_tensor.reshape((self.batch_size, np.prod(self.input_shape)))
        return flatten

    def backward(self, error_tensor):
        # treat batch size as tuple
        shape = (self.batch_size,) + self.input_shape
        tensor = error_tensor.reshape(shape)
        return tensor


