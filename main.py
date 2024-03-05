from environment import Environment
from robot import Robot
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors  

from matplotlib.widgets import Slider

# Define constants
MOTOR_NOISE_STD = 0.2
GPS_NOISE_STD = 2.5
IMU_NOISE_STD = 0.2

FACTORY_SIZE = 50
NUM_POINTS = 5
OBSTACLE_COUNT = 30


START_POINT = np.array([[0*FACTORY_SIZE/2], [0*FACTORY_SIZE/2]])
START_ORIENTATION = float(0)
INITIAL_STATE = START_POINT
INITIAL_COVARIANCE = np.array([[1, 0],[0, 1]])


# Instantiate classes
robot = Robot(START_POINT, START_ORIENTATION, MOTOR_NOISE_STD, GPS_NOISE_STD, IMU_NOISE_STD)
env = Environment(FACTORY_SIZE, NUM_POINTS, OBSTACLE_COUNT)

grid = env.generate_grid()



if __name__ == '__main__':
    # Set up the figure for animation again
        # Setup the figure and axis for the plot and sliders
    fig, ax = plt.subplots(figsize=(12, 10))  # You can adjust the figure size as needed
    plt.subplots_adjust(left=0.1, bottom=0.2, top=0.95)  # Adjust bottom to make room for sliders

    # Initialize an empty line for the A* path
    astar_path_line, = ax.plot([], [], 'g--', lw=1, label='A* Path')  # Green dashed line for A* path

    # Setup plot limits and initial plot elements
    ax.set_xlim((-1, FACTORY_SIZE))
    ax.set_ylim((-1, FACTORY_SIZE))
    line, = ax.plot([], [], 'o-', lw=2)
        # Display the grid with obstacles
    # Normalize grid values to [0,1] for displaying as an image
    ax.imshow(grid.T, origin='lower', cmap='gray', extent=[-1, FACTORY_SIZE, -1, FACTORY_SIZE], alpha=0.5)
    # Initialize an empty target marker
    
    target, = ax.plot([], [], 'rx', markersize=10)  # Empty plot for the target marker
        # Use a consistent aspect ratio
    ax.set_aspect('equal')

    # Add grid lines for better readability
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    # Add legends and improve their appearance
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    # Refine markers and lines
    line.set_linestyle('--')  # Set the robot's path to dashed
    line.set_color('blue')  # Set a clear color for the robot's path
    line.set_linewidth(2)  # Increase the width for visibility

    target.set_markersize(12)  # Increase the marker size for the target
    target.set_markerfacecolor('green')  # Change the target marker color

    # Use a different colormap for obstacles
    # For example, you can use 'gray' for free space and 'darkred' for obstacles
    obstacle_cmap = matplotlib.colors.ListedColormap(['gray', 'darkred'])
    ax.imshow(grid.T, origin='lower', cmap=obstacle_cmap, extent=[-1, FACTORY_SIZE, -1, FACTORY_SIZE], alpha=0.8)


    ax.set_title('Robot Navigation Simulation')
    axcolor = 'lightgoldenrodyellow'
    # Adjust slider sizes and positions
    slider_width = 0.75  # Width of the slider
    slider_height = 0.02  # Height of the slider
    vertical_gap = 0.008  # Vertical gap between sliders
    slider_start_bottom = 0.01  # Starting position of the bottom slider

    # Create sliders with adjusted positions
    ax_noise_motor = plt.axes([0.1, slider_start_bottom, slider_width, slider_height], facecolor=axcolor)
    ax_noise_gps = plt.axes([0.1, slider_start_bottom + slider_height + vertical_gap, slider_width, slider_height], facecolor=axcolor)
    ax_noise_imu = plt.axes([0.1, slider_start_bottom + 2 * (slider_height + vertical_gap), slider_width, slider_height], facecolor=axcolor)

    # Similarly adjust the positions for PID gain sliders
    ax_kp = plt.axes([0.1, slider_start_bottom + 3 * (slider_height + vertical_gap), slider_width, slider_height], facecolor=axcolor)
    ax_ki = plt.axes([0.1, slider_start_bottom + 4 * (slider_height + vertical_gap), slider_width, slider_height], facecolor=axcolor)
    ax_kd = plt.axes([0.1, slider_start_bottom + 5 * (slider_height + vertical_gap), slider_width, slider_height], facecolor=axcolor)


    s_noise_motor = Slider(ax_noise_motor, 'Motor Noise Std', 0.0001, 1.0, valinit=robot.motor_noise_std)
    s_noise_gps = Slider(ax_noise_gps, 'GPS Noise Std', 0.0001, 10.0, valinit=robot.gps_noise_std)
    s_noise_imu = Slider(ax_noise_imu, 'IMU Noise Std', 0.0001, 10.0, valinit=robot.imu_noise_std)

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
        true_line.set_data([], [])
        z_line.set_data([], [])
        return line, true_line, z_line

    # Initialize lists to store positions
    xdata, ydata = [], []
    true_xdata, true_ydata = [], []  # Lists to store true positions
    z_xdata, z_ydata = [], []  # Lists to store measurements

    # Initialize plot lines for true positions and measurements
    true_line, = ax.plot([], [], 'ko', markersize=5, label='True Position')  # Black points for true position
    z_line, = ax.plot([], [], 'ro', markersize=5, label='Measurements')  # Red points for measurements
    # Initialize target position with None or an initial value
    target_pos = None

    # Function to handle mouse click events
    def onclick(event):
        global target_pos, path, current_waypoint_index
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < FACTORY_SIZE and 0 <= y < FACTORY_SIZE and grid[y, x] == 0:
                start_pos = (int(robot.x_m[0, 0]), int(robot.x_m[1, 0]))  # Get robot's current position
                target_pos = np.array([[x], [y]])
                path = env.find_path(start_pos, (x, y), grid)  # Find the new path
                current_waypoint_index = 0  # Reset the waypoint index for the new path
                
                # Update the A* path line data
                if path:
                    astar_path_line.set_data(*zip(*path))
                else:
                    astar_path_line.set_data([], [])
                    
                target.set_data([x], [y])
                fig.canvas.draw()
            else:
                print("Clicked on an obstacle or outside the grid.")

    # Connect the click event handler to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)

    current_waypoint_index = 0  # Initialize a waypoint index
    path = []
    def animate(i):
        global target_pos, xdata, ydata, true_xdata, true_ydata, z_xdata, z_ydata, path, current_waypoint_index
    
        if path and current_waypoint_index < len(path):
            next_waypoint = path[current_waypoint_index]
            target_pos = np.array([[next_waypoint[0]], [next_waypoint[1]]])
            
            if np.linalg.norm(robot.x_m[:2] - target_pos) < 0.5:  # Check if close to the current waypoint
                current_waypoint_index += 1  # Proceed to the next waypoint

        # Move the robot towards the current waypoint
        if current_waypoint_index < len(path):
            robot.step_run(target_pos)

        # Update robot's trail and measurements on the plot
        xdata.append(robot.x_m[0, 0])
        ydata.append(robot.x_m[1, 0])
        true_xdata.append(robot.true_x[0, 0])
        true_ydata.append(robot.true_x[1, 0])
        z_xdata.append(robot.z[0, 0])
        z_ydata.append(robot.z[1, 0])

        line.set_data(xdata, ydata)
        true_line.set_data(true_xdata, true_ydata)
        z_line.set_data(z_xdata, z_ydata)

        return line, true_line, z_line, astar_path_line


    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=1000, interval=200, blit=False)
    plt.show()


