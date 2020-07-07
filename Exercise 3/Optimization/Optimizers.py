import numpy as np
from Optimization import Constraints


class Optimizer:
    def __init__(self):
        self.regularizer = None
        self.learning_rate = 1

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        self.regularizer.lr = self.learning_rate

class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def add_regularizer(self, regularizer):
        super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        weights = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weights = weights - self.regularizer.calculate_gradient(weight_tensor)
        return weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.vk = 0

    def add_regularizer(self, regularizer):
        super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.momentum_rate * self.vk - self.learning_rate * gradient_tensor
        weights = weight_tensor + self.vk

        if self.regularizer is not None:
            weights = weights - self.regularizer.calculate_gradient(weight_tensor)

        return weights


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()

        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        # k
        self.k = 0

        # list for v
        self.v = [0]

        # list for r
        self.r = [0]

    def add_regularizer(self, regularizer):
        super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1

        vk = self.mu * self.v[self.k - 1] + (1 - self.mu) * gradient_tensor
        rk = self.rho * self.r[self.k - 1] + (1 - self.rho) * (gradient_tensor * gradient_tensor)

        # bias correction
        vk_hat = vk / (1 - self.mu ** self.k)
        rk_hat = rk / (1 - self.rho ** self.k)

        self.v.append(vk)
        self.r.append(rk)

        # denominator: rk_hat = 0
        eps = np.finfo(np.asarray(rk_hat).dtype).eps

        w_update = weight_tensor - self.learning_rate * vk_hat / (np.sqrt(rk_hat) + eps)

        if self.regularizer is not None:
            w_update = w_update - self.regularizer.calculate_gradient(weight_tensor)

        return w_update
