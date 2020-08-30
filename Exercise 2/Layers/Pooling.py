
import numpy as np
# import skimage.measure

class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.input_shape = None
        self.max=[]

    def forward(self, input_tensor):

        self.input_shape = shape = np.shape(input_tensor)
        # list -> either y or y and x dimension
        input_size = list(shape[2:])
        # reset list for max coordinates
        self.max=[]


        if len(shape) == 3:
            strides = [0]
            # compute number of strides
            strides[0] = 1 + (input_size[0] - 1) // self.stride_shape[0]
            # check whether last stride fills full pool or stride
            if (strides[0] * max(self.pooling_shape[0], self.stride_shape[0])) > input_size[0]:
                strides[0] = strides[0] - 1
        elif len(shape) == 4:
            strides = [0,0]
            strides[0] = 1 + (input_size[0] - 1) // self.stride_shape[0]
            if (strides[0] * max(self.pooling_shape[0], self.stride_shape[0])) > input_size[0]:
                strides[0] = strides[0] - 1
            strides[1] = 1 + (input_size[1] - 1) // self.stride_shape[1]
            if (strides[1] * max(self.pooling_shape[1], self.stride_shape[1])) > input_size[1]:
                strides[1] = strides[1] - 1
        out = np.zeros((shape[0], shape[1], strides[0], strides[1]))

        # loop for every element of the batch
        for batch in range(shape[0]):
            temp = input_tensor

            # number of kernels determines output depth -> stack kernels
            for kernel in range(shape[1]):

                if len(shape) == 3:
                    for y in range(strides[0]):
                        y_pool_1 = y*self.stride_shape[0]
                        y_pool_2 = y*self.stride_shape[0]+self.pooling_shape[0]
                        pool = temp[batch,kernel,y_pool_1:y_pool_2]

                        argmax = np.argmax(pool)
                        self.max.append((batch, kernel, argmax))
                        out[batch, kernel, strides[0]] = pool[argmax]

                elif len(shape) == 4:
                    for y in range(strides[0]):
                        for x in range(strides[1]):
                            y_pool_1 = y*self.stride_shape[0]
                            y_pool_2 = y*self.stride_shape[0]+self.pooling_shape[0]
                            x_pool_1 = x*self.stride_shape[1]
                            x_pool_2 = x*self.stride_shape[1]+self.pooling_shape[1]
                            pool = temp[batch,kernel,y_pool_1:y_pool_2,x_pool_1:x_pool_2]

                            argmax = np.argmax(pool)
                            arg_y = argmax // self.pooling_shape[1]
                            arg_x = argmax % self.pooling_shape[1]
                            self.max.append((batch, kernel, arg_y+y_pool_1, arg_x+x_pool_1))
                            xy = pool[arg_y,arg_x]
                            out[batch,kernel,y,x] = xy
        return out

    def backward(self, error_tensor):
        out = np.zeros(self.input_shape)
        errors =np.ravel(error_tensor)
        i = 0
        for coord in self.max:
            temp = np.zeros(self.input_shape)
            temp[coord] = errors[i]
            out = out + temp
            i = i+1
        return out

