import numpy as np



class CrossEntropyLoss:

    # constructor
    def __init__(self):
        self.y_pred = None
        self.epsilon = np.finfo(float).eps

    # L = -y * ln(yhat)
    def forward(self, input_tensor, label_tensor):
        self.y_pred = input_tensor
        loss = -np.sum(np.log(input_tensor + self.epsilon) * label_tensor)
        #loss = -np.sum(np.dot(np.log(input_tensor + self.epsilon), label_tensor.T ))
        return loss

    # en = -y / yhat
    def backward(self, label_tensor):
        e = -np.true_divide(label_tensor, self.y_pred)
        return e
