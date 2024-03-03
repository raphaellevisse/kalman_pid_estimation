import numpy as np


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.target = self.random_target()

    def random_target(self):
        return np.array([[np.random.uniform(0, self.width)], [np.random.uniform(0, self.height)]])
    



