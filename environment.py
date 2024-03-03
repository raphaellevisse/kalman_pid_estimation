import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, grid_size, num_points):
        self.grid_size = grid_size
        self.num_points = num_points
        self.points = self.generate_points()
    
    def generate_points(self):
        # Generates random points within the grid
        points = np.random.rand(self.num_points, 2) * self.grid_size
        return points
    
    def plot_points(self, ax):
        # Plots the generated points on the provided axes
        ax.scatter(self.points[:, 0], self.points[:, 1], c='green', marker='x')
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.grid(True)
    



