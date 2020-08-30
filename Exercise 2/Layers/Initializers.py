
import numpy as np


class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in=1, fan_out=1):
        weights = self.constant * np.ones(weights_shape)
        return weights


class UniformRandom:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in=1, fan_out=1):
        weights = np.random.uniform(size=weights_shape)
        return weights


class Xavier:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / (fan_out + fan_in))
        weights = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return weights


class He:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0 / fan_in)
        weights = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return weights

