import numpy as np


class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        w = np.ones_like(weights_shape)
        w = self.constant * w
        return w


class UniformRandom:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        w = np.random.uniform(0, 1, weights_shape)
        return w


class Xavier:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        w = np.random.normal(0, sigma, weights_shape)
        return w


class He:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        w = np.random.normal(0, sigma, weights_shape)
        return w

