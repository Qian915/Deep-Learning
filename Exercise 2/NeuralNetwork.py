import copy
import numpy as np
from Layers import *
from Optimization import *


class NeuralNetwork:

    # constructor
    def __init__(self, optimizer, weights_initializer, bias_initializer):

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

        # create public members
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        # store training data
        self.input = None
        self.label = None
        self.input_dim = 0


    def forward(self, test=False):
        if test == False:
            # training data
            self.input, self.label = self.data_layer.forward()
        input_tensor = self.input

        # traverse every layer
        for layer in self.layers:
            # output of the former layer = input of the next layer
            input_tensor = layer.forward(input_tensor)

        if test == False:
            input_tensor = self.loss_layer.forward(input_tensor, self.label)

        return input_tensor

    def backward(self):
        # backpropagation from Loss layer
        error = self.loss_layer.backward(self.label)

        # backpropagation for every layer from back to front
        for layer in self.layers[::-1]:
            error = layer.backward(error)

        return error

    def append_trainable_layer(self, layer):
        # set initializer for the layer
        layer.initilize(self.weights_initializer, self.bias_initializer)

        # set optimizer for the layer
        optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer = optimizer

        # append the layer to layers
        self.layers.append(layer)

    def train(self, iterations):
        for epoch in range(iterations):
            y_pred = self.forward()
            input = self.backward()

            # store loss for each iteration
            self.loss.append(self.loss_layer.forward(self.label, y_pred))

    def test(self, input_tensor):
        self.input = input_tensor
        y_pred = self.forward(test=True)
        return y_pred










