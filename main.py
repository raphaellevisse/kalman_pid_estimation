from environment import Environment
from robot import Robot
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Define constants
MOTOR_NOISE_STD = 1.0
GPS_NOISE_STD = 2.0
IMU_NOISE_STD = 2.0

FACTORY_SIZE = 100
NUM_POINTS = 20

START_POINT = np.array([[0], [0]])
START_ORIENTATION = float(0)
INITIAL_STATE = START_POINT
INITIAL_COVARIANCE = np.array([[1, 0],[0, 1]])


# Instantiate classes
robot = Robot(START_POINT, START_ORIENTATION, MOTOR_NOISE_STD, GPS_NOISE_STD, IMU_NOISE_STD)
env = Environment(FACTORY_SIZE, NUM_POINTS)





if __name__ == '__main__':
        # Define a function to update the plot
    def update_plot(ax, true_positions, estimated_positions, gps_positions, target_pos):
        # Plot true, estimated, and GPS positions
        ax.plot(np.array(true_positions)[:, 0], np.array(true_positions)[:, 1], 'k-', label='True Position' if len(ax.lines) == 0 else "")
        ax.plot(np.array(estimated_positions)[:, 0], np.array(estimated_positions)[:, 1], 'b--', label='Kalman Filter Estimate' if len(ax.lines) == 0 else "")
        ax.plot(np.array(gps_positions)[:, 0], np.array(gps_positions)[:, 1], 'r:', label='Unfiltered Estimate' if len(ax.lines) == 0 else "")
        ax.plot(target_pos[0], target_pos[1], 'go', label='Target' if len(ax.lines) == 0 else "")

        # If it's the first time, add the legend
        if len(ax.lines) == 4:  # Only add legend once
            ax.legend()
            ax.grid(True)

    # Set up the plot once
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Navigation with Kalman Filter')
    env.plot_points(ax=ax)  # Plot the points

    # Run the robot for each target position and update the plot
    for target in env.points:
        print("target is: ", target)
        target_pos = target.reshape(2, 1)
        estimated_positions, true_positions, gps_positions = robot.run(target_pos=target_pos, steps=500, plot=False)
        update_plot(ax, true_positions, estimated_positions, gps_positions, target_pos)

    
    # Gather current limits
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # Gather all x and y data from the plot
    x_data = []
    y_data = []
    for line in ax.lines:
        x_data.extend(line.get_xdata())
        y_data.extend(line.get_ydata())

    # Calculate new limits
    buffer = 1  # or any other value you find appropriate
    new_xlim = [min(x_data) - buffer, max(x_data) + buffer]
    new_ylim = [min(y_data) - buffer, max(y_data) + buffer]

    # Update the plot limits if they are exceeded
    if new_xlim[0] < current_xlim[0]:
        current_xlim = (new_xlim[0], current_xlim[1])
    if new_xlim[1] > current_xlim[1]:
        current_xlim = (current_xlim[0], new_xlim[1])
    if new_ylim[0] < current_ylim[0]:
        current_ylim = (new_ylim[0], current_ylim[1])
    if new_ylim[1] > current_ylim[1]:
        current_ylim = (current_ylim[0], new_ylim[1])

    ax.set_xlim(current_xlim)
    ax.set_ylim(current_ylim)

    # Show the updated plot
    plt.show()
    # Other imports and robot/environment setup as before

    def animate(i):
        # Clear the current axes.
        ax.clear()
        # Optionally, re-plot environment points or other static elements here
        env.plot_points(ax=ax)  # Assuming this doesn't change
        
        # Update the plot for the i-th step
        # You might need to access pre-computed step data here
        update_plot(ax, true_positions[i], estimated_positions[i], gps_positions[i], env.points[i])
        
        # Update plot limits or perform other necessary updates

    # Initial plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Navigation with Kalman Filter')

    # Prepare data for animation
    # This might involve running the robot simulation and storing results in lists or arrays

    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(env.points), interval=100)

    plt.show()
        

