import numpy as np


class SoftMax:

    # constructor
    def __init__(self):
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
        #print(np.sum(out, axis=1))
        self.y_pred = out

        return out

    # E(n-1) = yhat * (En - sum(En,j * yj_hat))
    def backward(self, error_tensor):
        prod = error_tensor * self.y_pred
        sum = np.sum(prod, axis=1, keepdims=True)
        #sum = np.expand_dims(sum, axis=1)
        out = self.y_pred * (error_tensor - sum)

        #s = self.y_pred
        #sisj = np.matmul(np.expand_dims(s,axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        #error_tensor_exp = np.expand_dims(error_tensor, axis=1)
        #tmp = np.matmul(error_tensor_exp, sisj) #(N, 1, c)
        #tmp = np.squeeze(tmp, axis=1)
        #tmp = -tmp + error_tensor * s
        return out
