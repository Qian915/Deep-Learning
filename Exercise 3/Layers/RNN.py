import numpy as np
from Layers.Base import BaseLayer
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected
from Layers import Initializers


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # hidden layer & output layer for RNN
        self.fc_h = FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.fc_y = FullyConnected(self.hidden_size, self.output_size)

        # initialize hidden_state with all 0
        self.hidden_state = []
        self.hidden_state.append(np.zeros(hidden_size))  # a vector!

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
        self.t = input_tensor.shape[0]
        output = []
        for t in range(input_tensor.shape[0]):
            if not self.memorize and t > 0:
                self.hidden_state[t-1] = np.zeros(self.hidden_size)

            # ht
            x_composed = np.hstack((input_tensor[t], self.hidden_state[t-1 if t > 0 else 0]))
            if t == 0:
                self.hidden_state[t] = TanH().forward(self.fc_h.forward(x_composed))
            else:
                self.hidden_state.append(TanH().forward(self.fc_h.forward(x_composed)))

            # yt
            output.append(self.fc_y.forward(self.hidden_state[t]))
            output[t] = Sigmoid().forward(output[t])

        result = np.array(output)
        # store activations for sigmoid & tanh layer
        self.sigmoid_activations = result
        self.tanh_activations = np.array(self.hidden_state)

        return result

    def backward(self, error_tensor):
        sigmoid = Sigmoid()
        tanh = TanH()
        gradient_wy = None
        gradient_wh = None
        gradient_ht = np.zeros((self.t, self.hidden_size))
        for t in range(error_tensor.shape[0]-1, -1, -1):
            # gradient w.r.t. W_y
            # set activations for sigmoid layer
            sigmoid.activations = self.sigmoid_activations[t]
            gradient_ot = sigmoid.backward(error_tensor[t])

            # gradient w.r.t. ht
            # t = 0 / T: only one part of sum
            if t == error_tensor.shape[0] - 1:
                gradient_ht[t] = self.fc_y.backward(gradient_ot)

            else:
                # set activations for tanh layer
                tanh.activations = self.tanh_activations[t+1]
                gradient_ut = tanh.backward(gradient_ht[t+1])
                # decompose Wh: W_xh, W_hh
                wh = self.fc_h.backward(gradient_ut)
                if t == 0:
                    gradient_ht[t] = wh[:, self.input_size:self.input_size+self.hidden_size]
                else:
                    gradient_ht[t] = wh[:, self.input_size:self.input_size+self.hidden_size] + self.fc_y.backward(gradient_ot)

            # gradient w.r.t. W_y
            gradient_wy += self.fc_y.gradient_weights

            # gradient w.r.t W_h
            tanh.activations = self.tanh_activations[t]
            error = self.fc_h.backward(tanh.backward(gradient_ht[t]))
            gradient_wh += self.fc_h.gradient_weights

            # decompose Wh: W_xh, W_hh
            # gradient_whh = gradient_wh[self.input_size:self.input_size+self.hidden_size, :]
            # gradient_wxh = gradient_wh[0:self.input_size, :]

        return error

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size+self.hidden_size, self.output_size), self.input_size+self.hidden_size, self.output_size)
        self.bias = bias_initializer.initialize(self.output_size, self.input_size+self.hidden_size, self.output_size)

        self.weights = np.vstack((self.weights, self.bias))
