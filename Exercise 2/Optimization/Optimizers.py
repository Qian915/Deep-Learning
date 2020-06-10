import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - np.dot(self.learning_rate, gradient_tensor)


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        # list for movement
        self.v = [0]
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1
        vk = self.momentum_rate * self.v[self.k - 1] - self.learning_rate * gradient_tensor
        self.v.append(vk)

        w_update = weight_tensor + vk
        return w_update


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        # k
        self.k = 0

        # list for v
        self.v = [0]

        # list for r
        self.r = [0]

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
        return w_update







