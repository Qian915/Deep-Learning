import copy
import numpy as np
from Layers import *
from Optimization import *


class NeuralNetwork:

    # create public members
    loss = []
    layers = []
    data_layer = None
    loss_layer = None

    # store training data
    input = None
    label = None
    input_dim = 0

    # constructor
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def forward(self):
        # training data
        self.input, self.label = self.data_layer.forward()
        input_tensor = self.input

        # traverse every layer
        for layer in self.layers:
            # output of the former layer = input of the next layer
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    def backward(self):
        # backpropagation from Loss layer
        error = self.loss_layer.backward(self.label)

        # backpropagation for every layer from back to front
        for layer in self.layers[::-1]:
            error = layer.backward(error)

        return error

    def append_trainable_layer(self, layer):
        optimizer = copy.deepcopy(self.optimizer)

        # set optimizer for the layer
        layer.optimizer = optimizer

        # append the layer to layers
        self.layers.append(layer)

    def train(self, iterations):
        # initialization for weights
        w = np.zeros((self.input_dim,))

        for epoch in range(iterations):
            y_pred = self.forward()
            w_grad = self.backward()

            # weights update
            w = self.optimizer.calculate_update(w, w_grad)

            # store loss for each iteration
            loss.append(self.loss_layer.forward(y_pred, self.label))

    def test(self, input_tensor):
        self.input = input_tensor
        y_pred = self.forward()
        return y_pred









