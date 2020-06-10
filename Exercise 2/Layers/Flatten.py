class Flatten:

    def __init__(self):
        self.input = None
        self.error = None

    def forward(self, input_tensor):
        self.input = input_tensor
        # input for FC: (batch_size, input_size)
        out = input_tensor.reshape(input_tensor.shape[0], -1)
        return out

    def backward(self, error_tensor):
        # error_tensor: (batch_size, input_size)
        out = error_tensor.reshape(self.input.shape)
        return out
