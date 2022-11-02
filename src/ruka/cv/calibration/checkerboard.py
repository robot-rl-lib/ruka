import numpy as np

class Checkerboard:
    def __init__(self, size, cell_size):
        self.size = size
        self.cell_size = cell_size

    def get_points(self):
        points = []
        for i in range(self.size[1]):
            for j in range(self.size[0]):
                points.append([i * self.cell_size, j * self.cell_size, 0])
        return np.array(points)
