import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Checker:
    output = np.ndarray
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        if(self.resolution % (2 * self.tile_size) == 0):
            #checkerboard
            rep_times = self.resolution / self.tile_size / 2
            base = np.array([[1,0], [0,1]])
            self.output = np.tile(base, (rep_times, rep_times))
            copy = self.output
            return copy

    def show(self):
        #color
        checkerboard_color = LinearSegmentedColormap.from_list('checkerboard_cmap', [(0,0,0), (1,1,1)])
        plt.imshow(self.output, cmap=checkerboard_color)
        plt.show()

class Circle:

