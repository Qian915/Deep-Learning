import numpy as np
from Layers.Base import BaseLayer
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected
from Layers import Sigmoid
from Layers import Initializers


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # hidden layer & output layer for RNN cell
        self.fc_h = FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.fc_y = FullyConnected(self.hidden_size, self.output_size)

        # initialize hidden_state with all 0
        self.hidden_state = []
        self.hidden_state.append(np.zeros((1, hidden_size)))

        self.output = []
        self._memorize = False
        self._gradient_weights = None
        self._optimizer = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, new_memorize):
        self._memorize = new_memorize

    @property
    def weights(self):
        return self.fc_h.weights

    @weights.setter
    def weights(self, new_weights):
        self.fc_h.weights = new_weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, new_grad_w):
        self._gradient_weights = new_grad_w

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    def forward(self, input_tensor):
        for t in range(input_tensor.shape[0]):
            if not self.memorize and t > 0:
                self.hidden_state[t-1] = np.zeros((1, self.hidden_size))
            # ht
            x_composed = np.hstack((input_tensor[t], np.array(self.hidden_state[t-1 if t > 0 else 0])))
            self.hidden_state[t] = TanH.forward(self.fc_h.forward(x_composed))

            # yt
            self.output.append(Sigmoid(self.fc_y.forward(self.hidden_state[t])))

        self.hidden_state = np.array(self.hidden_state)
        return np.array(self.output)

    def backward(self, error_tensor):
        # gradient w.r.t. x
        # !!!!! not finished !!!!!
        for t in range(error_tensor.shape[0]-1, -1, -1):
            error_h = self.fc_y.backward(Sigmoid.backward(error_tensor))
            gradient_x_composed = self.fc_h.backward(TanH.backward(error_h))
            # split Xt, ht-1
            error = gradient_x_composed[:, 0:self.input_size]

        return error

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size+self.hidden_size, self.output_size), self.input_size+self.hidden_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), self.input_size+self.hidden_size, self.output_size)

        self.weights = np.vstack((self.weights, self.bias))
