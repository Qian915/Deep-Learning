import numpy as np
from Layers.Base import BaseLayer
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected
from Layers import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # initialize hidden_state with all 0
        self.hidden_state = np.zeros((1, hidden_size))
        self.output = None
        self.memorize = False
        self.weights = None
        self.gradient_weights = None
        self.optimizer = None

    @property
    def memorize(self):
        return self.memorize

    @memorize.setter
    def memorize(self, new_memorize):
        self.memorize = new_memorize

    @property
    def weights(self):
        return self.weights

    @weights.setter
    def weights(self, new_weights):
        self.weights = new_weights

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, new_grad_w):
        self.gradient_weights = new_grad_w

    @property
    def optimizer(self):
        return self.optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self.optimizer = new_optimizer

    def forward(self, input_tensor):
        for t in range(input_tensor.shape[0]):
            if not self.memorize:
                self.hidden_state = np.zeros((1, self.hidden_size))
            # ht
            x_composed = np.concatenate((input_tensor[t], self.hidden_state, np.ones(1)))
            input_size = x_composed.shape[0]
            output_size = self.output_size
            fc_h = FullyConnected(input_size, output_size)
            self.hidden_state = TanH.forward(fc_h.forward(x_composed))
            self.weights = fc_h.weights

            # yt
            input_size = self.hidden_state.shape[0]
            fc_y = FullyConnected(input_size, output_size)
            self.output = np.vstack(self.output, Sigmoid(fc_y.forward(self.hidden_state)))

        return self.output

    def backward(self, error_tensor):
        pass
