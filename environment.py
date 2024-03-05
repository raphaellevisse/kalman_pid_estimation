import numpy as np
import matplotlib.pyplot as plt
import heapq

class Environment:
    def __init__(self, grid_size, num_points, obstacle_count):
        self.grid_size = grid_size
        self.num_points = num_points
        self.points = self.generate_points()
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.obstacle_count = obstacle_count

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

    def generate_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for _ in range(self.obstacle_count):
            # Randomly place rectangular obstacles
            x, y = np.random.randint(0, self.grid_size, 2)
            width, height = np.random.randint(10, 100, 2)
            grid[max(0, x-width//2):min(self.grid_size, x+width//2), max(0, y-height//2):min(self.grid_size, y+height//2)] = 1
        return grid
    
    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def find_neighbors(self, grid, node):
        """Find neighbors of the current node that are not obstacles, including diagonals"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),  # 4-way connectivity
                    (1, 1), (1, -1), (-1, -1), (-1, 1)]  # Diagonals
        neighbors = []
        for direction in directions:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] == 0:  # Check if the neighbor is not an obstacle
                    neighbors.append(neighbor)
        return neighbors
    
    def find_path(start, goal, grid):
        """Find path from start to goal using A* with diagonal movement allowed"""
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[2]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in find_neighbors(grid, current):
                # Adjust movement cost based on whether the move is diagonal
                if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                    movement_cost = np.sqrt(2)  # Diagonal movement cost
                else:
                    movement_cost = 1  # Straight movement cost
                    
                tentative_g_score = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                    
        return []




                
            

        



