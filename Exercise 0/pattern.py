import numpy as np
import matplotlib.pyplot as plt

class Checker:
    output = np.ndarray

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        if(self.resolution % (2 * self.tile_size) == 0):
            
            #adjustable size of checkerboard
            copy_times = int(self.resolution / (self.tile_size * 2))
            
            #one tile
            black_tile = np.zeros((self.tile_size, self.tile_size))
            white_tile = np.ones((self.tile_size, self.tile_size))
            
            # one base checkerboard
            col1 = np.vstack((black_tile, white_tile))
            col2 = np.vstack((white_tile, black_tile))
            base_checkerboard = np.hstack((col1, col2))
            
            # checkerboard
            self.output = np.tile(base_checkerboard, (copy_times, copy_times))
            copy = self.output
            return copy

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()



class Circle:
    output = np.ndarray
    
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # matrix of x coordinate
        x = np.arange(self.resolution)
        x = np.expand_dims(x, axis=0)
        x = np.tile(x, (self.resolution, 1))

        # matrix of y coordinate
        y = np.rot90(x, -1)

        # matrix of circle
        circle = (x - self.position[0]) * (x - self.position[0]) + (y - self.position[1]) * (y - self.position[1])
        circle = circle <= self.radius * self.radius

        self.output = circle
        copy = self.output
        return copy

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()

# R - Red
# G - Green
# B - Blue
class Spectrum:
    # Array for generated pattern
    output = np.ndarray

    # set initial constructor values
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        # create 1d-array of resolution length
        horizontal = np.arange(0, self.resolution, 1)
        # reshape to horizontal 2d-array
        horizontal = np.reshape(horizontal, (self.resolution, 1))

        # create 1d-array of ones
        ones = np.ones(self.resolution)
        # reshape to vertical 2d-array
        ones = np.reshape(ones, (1, self.resolution))
        # create 2d-image by multiplying arrays
        dir1 = horizontal * ones
        #  rescale maximum value to 1
        dir1 = dir1 / np.max(dir1)
        # rotate image by 90 degrees
        dir2 = np.rot90(dir1)
        dir3 = np.rot90(dir2)
        dir4 = np.rot90(dir3)

        # reshape images to 3d-volumes
        dir2 = np.reshape(dir2, (self.resolution, self.resolution, 1))
        dir1 = np.reshape(dir1, (self.resolution, self.resolution, 1))
        dir3 = np.reshape(dir3, (self.resolution, self.resolution, 1))
        dir4 = np.reshape(dir4, (self.resolution, self.resolution, 1))

        # try and error to find out volume combination for rgb example in exercise
        # stack images on each other to create and return rgb-volume
        rgb = np.concatenate((dir2, dir1, dir4), 2)
        self.output = rgb
        return np.ndarray.copy(self.output)

    # plot and show pattern
    def show(self):
        plt.imshow(self.output)
        plt.show()
