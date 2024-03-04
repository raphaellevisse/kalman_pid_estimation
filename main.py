from environment import Environment
from robot import Robot
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

# Define constants
MOTOR_NOISE_STD = 0.6
GPS_NOISE_STD = 0.8
IMU_NOISE_STD = 0.2

FACTORY_SIZE = 100
NUM_POINTS = 5

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
    positions = [np.array([0, 0])]
    # Run the robot for each target position and update the plot
    
    for target in env.points:
        print("target is: ", target)
        target_pos = target.reshape(2, 1)
        estimated_positions, true_positions, gps_positions = robot.run(target_pos=target_pos, steps=500, plot=False)
        positions = positions + estimated_positions
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
    

    # Set up the figure for animation again
        # Setup the figure and axis for the plot and sliders
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)  # Adjust to make room for sliders  # Adjust to make room for sliders

    # Setup plot limits and initial plot elements
    ax.set_xlim((-1, 100))
    ax.set_ylim((-1, 100))
    line, = ax.plot([], [], 'o-', lw=2)

    # Initialize an empty target marker

    target, = ax.plot([], [], 'rx', markersize=10)  # Empty plot for the target marker

    axcolor = 'lightgoldenrodyellow'
    ax_noise_motor = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor= axcolor)
    ax_noise_gps = plt.axes([0.1, 0.00, 0.65, 0.03], facecolor= axcolor)
    ax_noise_imu = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor= axcolor)  # Place IMU noise slider after GPS noise slider

    s_noise_motor = Slider(ax_noise_motor, 'Motor Noise Std', 0.0001, 5.0, valinit=robot.motor_noise_std)
    s_noise_gps = Slider(ax_noise_gps, 'GPS Noise Std', 0.0001, 5.0, valinit=robot.gps_noise_std)
    s_noise_imu = Slider(ax_noise_imu, 'IMU Noise Std', 0.0001, 5.0, valinit=robot.imu_noise_std)

    # Update functions for noise sliders
    def update_noise_motor(val):
        robot.motor_noise_std = val
        robot.Q = np.diag([val**2, val**2, val**2])  # Update the process noise covariance matrix accordingly

    def update_noise_gps(val):
        robot.gps_noise_std = val
        robot.R[0, 0] = val**2
        robot.R[1, 1] = val**2  # Assuming you want to update GPS noise std in R matrix

    def update_noise_imu(val):
        robot.motor_noise_std = val
        robot.R[2,2] = val**2

    s_noise_motor.on_changed(update_noise_motor)
    s_noise_gps.on_changed(update_noise_gps)
    s_noise_imu.on_changed(update_noise_imu)
                           
        
    # Add sliders for PID gains
    ax_kp = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_ki = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_kd = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)  # Adjust positions as necessary

    s_kp = Slider(ax_kp, 'Kp', 0.0, 10.0, valinit=robot.Kp_rot)
    s_ki = Slider(ax_ki, 'Ki', 0.0, 10.0, valinit=robot.Ki_rot)
    s_kd = Slider(ax_kd, 'Kd', 0.0, 10.0, valinit=robot.Kd_rot)
    # Slider update functions
    def update_kp(val):
        robot.Kp_rot = val
    s_kp.on_changed(update_kp)

    def update_ki(val):
        robot.Ki_rot = val
    s_ki.on_changed(update_ki)

    def update_kd(val):
        robot.Kd_rot = val
    s_kd.on_changed(update_kd)

    # Initialize function for the animation
    def init():
        line.set_data([], [])
        return (line,)

    # Initialize lists to store positions
    xdata, ydata = [], []

        # Initialize target position with None or an initial value
    target_pos = None

    # Function to handle mouse click events
    def onclick(event):
        global target_pos
        print(event)
        if event.xdata is not None and event.ydata is not None and 0 <= event.xdata <= 100 and 0 <= event.ydata <= 100:
            # Update target_pos with the coordinates of the click event
            target_pos = np.array([[event.xdata], [event.ydata]])
            # Update the target marker to the new position
            target.set_data([event.xdata], [event.ydata])
            print(f"New target position: {target_pos.T}")
        else:
            print("Click was outside the grid. No action taken.")
    
    # Connect the click event handler to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)

    def animate(i):
        global target_pos, xdata, ydata

        # Check if a target position has been set by a user click
        if target_pos is not None:
            # Append current position to the lists
            xdata.append(robot.x_m[0, 0])
            ydata.append(robot.x_m[1, 0])

            # Check if the robot is close enough to the target position
            if np.sqrt((robot.x_m[0, 0] - target_pos[0, 0])**2 + (robot.x_m[1, 0] - target_pos[1, 0])**2) > 0.5:
                # If not, continue moving the robot towards the target position
                robot.step_run(target_pos)
            else:
                # Optionally, you can handle what happens once the target is reached here
                # For example, you might clear the target_pos or take some other action
                print("Reached the target position.")
                # target_pos = None  # Uncomment if you wish to clear the target after reaching it

        # Update the line with all previous positions
        line.set_data(xdata, ydata)
        return (line,)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(positions), interval=200, blit=False)
    plt.show()


