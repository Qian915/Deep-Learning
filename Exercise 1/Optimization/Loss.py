import numpy as np


class CrossEntropyLoss:

    # constructor
    def __init__(self):
        self.y_pred = None

    # L = -y * ln(yhat)
    def forward(self, input_tensor, label_tensor):
        self.y_pred = input_tensor

        # eps to prevent log(0)
        eps = np.finfo(input_tensor.dtype).eps
        input_tensor = np.maximum(input_tensor, eps)

        loss = np.sum(-np.log(input_tensor) * label_tensor)

        return loss

    # en = -y / yhat
    def backward(self, label_tensor):
        e = -np.divide(label_tensor, self.y_pred)
        return e
