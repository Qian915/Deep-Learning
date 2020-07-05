import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):

    # constructor
    def __init__(self):
        super().__init__()
        # store input vector
        self.input = None
        self.y_pred = None

        # initialize weights
        self.weights = None

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor)
        x_exp = np.exp(input_tensor)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / partition
        # print(np.sum(out, axis=1))
        self.y_pred = out

        return out

    # E(n-1) = yhat * (En - sum(En,j * yj_hat))
    def backward(self, error_tensor):
        prod = error_tensor * self.y_pred
        sum = np.sum(prod, axis=1, keepdims=True)
        # sum = np.expand_dims(sum, axis=1)
        out = self.y_pred * (error_tensor - sum)

        return out
