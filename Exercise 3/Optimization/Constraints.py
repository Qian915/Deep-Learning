import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.lr = 1

    def calculate_gradient(self, weights):
        return self.alpha * weights * self.lr

    def norm(self, weights):
        return self.alpha * np.linalg.norm(weights)


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.lr = 1

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights) * self.lr

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
