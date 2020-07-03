
import numpy as np
import copy
from scipy.signal import convolve
from scipy.signal import correlate

from Layers import Initializers


class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # !!! single value or tuple !!!
        self.stride_shape = stride_shape

        self.input_channels = convolution_shape[0]
        self.kernel_shape = convolution_shape[1:]
        self.num_kernels = num_kernels
        self.fan_in = convolution_shape
        self.fan_out = np.asarray(self.fan_in)
        self.fan_out[0] = num_kernels

        self.pad_x1 = self.pad_x2 = self.pad_y1 = self.pad_y2 = 0

        # set padding for y-axis
        if self.kernel_shape[0] % 2 == 0:
            self.pad_y1 = self.kernel_shape[0] // 2 - 1
            self.pad_y2 = self.kernel_shape[0] // 2
        else:
            self.pad_y1 = self.pad_y2 = self.kernel_shape[0] // 2

        # set padding for x-axis (if 2d convolution)
        if len(self.kernel_shape) == 2:
            if self.kernel_shape[1] % 2 == 0:
                self.pad_x1 = self.kernel_shape[1] // 2 - 1
                self.pad_x2 = self.kernel_shape[1] // 2
            else:
                self.pad_x1 = self.pad_x2 = self.kernel_shape[1] // 2

        # initialize weights
        self.weights = Initializers.UniformRandom().initialize(([self.num_kernels] + list(self.fan_in)), np.prod(self.fan_in), np.prod(self.fan_out))
        self.bias = Initializers.Constant(0.1).initialize((self.num_kernels, 1), np.prod(self.fan_in), np.prod(self.fan_out))

        self._gradient_weights = None
        self._gradient_bias = None

        #self.optimizer = None
        self._weight_optimizer = None
        self._bias_optimizer = None

        self.input_shape = None
        self.error_shape = None

    # property: gradient:weights, gradient_bias, optimizer
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
    @property
    def optimizer(self):
        return self._weight_optimizer
    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._weight_optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(self._weight_optimizer)

    def forward(self, input_tensor):
        self.original_input_tensor = input_tensor
        # store input shape for backward pass
        self.input_shape = np.shape(input_tensor)
        # sample shape for convenience
        self.unstrided_size = list(self.input_shape[2:])
        sample_size = list(self.input_shape[2:])

        # compute output sample size based on stride
        if len(self.input_shape) == 3:
            self.slice_y = slice(0, self.input_shape[2], self.stride_shape[0])
            sample_size[0] = 1 + (self.unstrided_size[0]-1) // self.stride_shape[0]
        elif len(self.input_shape) == 4:
            self.slice_y = slice(0, self.input_shape[2], self.stride_shape[0])
            self.slice_x = slice(0, self.input_shape[3], self.stride_shape[1])
            sample_size[0] = 1 + (self.unstrided_size[0] - 1) // self.stride_shape[0]
            sample_size[1] = 1 + (self.unstrided_size[1] - 1) // self.stride_shape[1]

        # empty array to store output tensor
        output = np.ndarray(tuple([0] + [self.num_kernels] + list(sample_size)))

        # loop for every element of the batch
        for batch in range(self.input_shape[0]):
            # fetch and reshape single sample of batch
            self.input = input_tensor[batch]
            self.input = self.input.reshape([1]+list(self.input_shape[1:]))

            # empty array to store kernel tensor
            kernels = np.ndarray((1, 0) + tuple(sample_size))

            # padding of sample tensor
            if len(self.input_shape) == 3:
                self.input = np.pad(self.input, ( (0, 0), (0, 0), (self.pad_y1, self.pad_y2) ) )
            elif len(self.input_shape) == 4:
                self.input = np.pad(self.input, ( (0, 0), (0, 0), (self.pad_y1, self.pad_y2), (self.pad_x1, self.pad_x2) ) )

            # number of kernels determines output depth -> stack kernels
            for kernel in range(self.num_kernels):
                # forward pass -> correlation
                weight = np.reshape(self.weights[kernel], ([1] + list(np.shape(self.weights[kernel]))))
                out = correlate(self.input, weight, mode='valid') + self.bias[kernel]

                # perform strided convolution by dropping unnecessary kernel layers
                if len(self.input_shape) == 3:
                    out = out[:, :, self.slice_y]
                elif len(self.input_shape) == 4:
                    out = out[:, :, self.slice_y, self.slice_x]

                # append convolution output to one kernel
                kernels = np.append(kernels, out, axis=1)
            # append kernel output to one sample
            output = np.append(output, kernels, axis=0)
        # return output tensor
        return output

    def backward(self, error_tensor):
        # shape of error tensor
        self.error_shape = np.shape(error_tensor)

        # reorder weights for backwards pass
        # empty array for backwards weights
        back_weight = np.ndarray(tuple([0] + [self.error_shape[1]] + list(np.shape(self.weights)[2:])))
        # loop over all input channels
        for input in range(self.input_shape[1]):
            # empty array to store weights of "backward kernels"
            temp_weight = np.ndarray(tuple([1] + [0] + list(np.shape(self.weights)[2:])))
            # loop over all gradient_layer kernels -> flip channel dimension
            for gradient_layer in range(self.error_shape[1]-1,-1,-1):
                # get and reshape single weight
                temp = np.reshape(self.weights[gradient_layer, input], ([1]+[1]+list(np.shape(self.weights)[2:])))
                # append weights to "backwards kernel"
                temp_weight = np.append(temp_weight, temp,axis=1)
            # append kernels to backwards weights tensor
            back_weight = np.append(back_weight, temp_weight, axis=0)

        # upsample error tensor (for strided convolution) awkward implementation :(
        if len(self.error_shape) == 3:
            upsampled_tensor = np.zeros((self.error_shape[0], self.error_shape[1], self.unstrided_size[0]))
            for ax0 in range(self.error_shape[0]):
                for ax1 in range(self.error_shape[1]):
                    i = 0
                    for ax2 in range(0, self.unstrided_size[0], self.stride_shape[0]):
                        upsampled_tensor[ax0, ax1, ax2] = error_tensor[ax0, ax1, i]
                        i = i+1
        elif len(self.error_shape) == 4:
            upsampled_tensor = np.zeros((self.error_shape[0], self.error_shape[1], self.unstrided_size[0], self.unstrided_size[1]))
            for ax0 in range(self.error_shape[0]):
                for ax1 in range(self.error_shape[1]):
                    j = 0
                    for ax2 in range(0, self.unstrided_size[0], self.stride_shape[0]):
                        i = 0
                        for ax3 in range(0, self.unstrided_size[1], self.stride_shape[1]):
                            upsampled_tensor[ax0, ax1, ax2, ax3] = error_tensor[ax0, ax1, j, i]
                            i = i+1
                        j = j+1

        # perform "backwards convolution"
        # empty array to store gradient_layer tensor
        gradient_layer = np.ndarray(tuple([0] + [self.input_shape[1]] + list(np.shape(upsampled_tensor)[2:])))
        # loop for every element of the batch
        for batch in range(self.error_shape[0]):
            # fetch and reshape single sample of batch
            input = upsampled_tensor[batch]
            input = input.reshape([1] + list(np.shape(upsampled_tensor)[1:]))
            # empty array to store kernel tensor
            kernels = np.ndarray((1, 0) + tuple(np.shape(upsampled_tensor)[2:]))

            # padding of sample tensor
            if len(self.error_shape) == 3:
                input = np.pad(input, ((0, 0), (0, 0), (self.pad_y1, self.pad_y2)))
            elif len(self.error_shape) == 4:
                input = np.pad(input, ((0, 0), (0, 0), (self.pad_y1, self.pad_y2), (self.pad_x1, self.pad_x2)))

            # number of input channels determines gradient_layer depth -> stack channels
            for kernel in range(self.input_shape[1]):
                # backward pass -> convolution
                weight = np.reshape(back_weight[kernel], ([1] + list(np.shape(back_weight[kernel]))))
                # forward pass correlation -> backward pass convolution
                out = convolve(input, weight, mode='valid')
                # append convolution gradient_layer for one kernel
                kernels = np.append(kernels, out, axis=1)
            # append kernel gradient_layer for one sample
            gradient_layer = np.append(gradient_layer, kernels, axis=0)

        # compute weights and bias gradient
        self.gradient_weights = np.zeros(np.shape(self.weights))
        self.gradient_bias = np.zeros(np.shape(self.bias))
        for batch in range(self.error_shape[0]):
            grad_weight = np.ndarray((0,) + tuple(self.fan_in[:]))

            input = self.original_input_tensor[batch]
            input = input.reshape([1] + list(np.shape(self.original_input_tensor)[1:]))

            # padding of sample tensor
            if len(np.shape(self.original_input_tensor)) == 3:
                input = np.pad(input, ( (0, 0), (0, 0), (self.pad_y1, self.pad_y2) ) )
            elif len(np.shape(self.original_input_tensor)) == 4:
                input = np.pad(input, ( (0, 0), (0, 0), (self.pad_y1, self.pad_y2), (self.pad_x1, self.pad_x2) ) )

            # number of error_kernels determines gradient_weight depth
            for kernel in range(self.error_shape[1]):
                # get kernel from error_tensor
                error = np.reshape(upsampled_tensor[batch, kernel], ([1,1] + list(np.shape(upsampled_tensor[batch, kernel]))))
                # forward pass 3d correlation -> backward pass 3d correlation
                out = correlate(input, error, mode='valid')
                # append grad_weight
                grad_weight = np.append(grad_weight, out, axis=0)
                self.gradient_bias[kernel] = self.gradient_bias[kernel] + np.sum(error)
            # sum gradient values for every sample in the batch
            self.gradient_weights = self.gradient_weights + grad_weight

        # update weights
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        # return gradient_layer tensor
        return gradient_layer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(([self.num_kernels] + list(self.fan_in)), np.prod(self.fan_in), np.prod(self.fan_out))
        self.bias = bias_initializer.initialize((self.num_kernels, 1), np.prod(self.fan_in), np.prod(self.fan_out))
