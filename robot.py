import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


np.random.seed(1) ## for repeatability

class Robot:
    def __init__(self, start_pos, start_orientation, motor_noise_std, gps_noise_std, imu_noise_std):
        # Defining state
        self.true_x = np.array([[start_pos[0, 0]], [start_pos[1, 0]], [np.radians(float(start_orientation))]])

        self.orientation_rad = np.radians(float(start_orientation))

        self.position = np.array(start_pos, dtype=float)  # Robot's position array [x, y].T
        
        # Kalman filter necessities
        self.A = np.eye(3)

        self.x_m = np.array([[start_pos[0, 0]], [start_pos[1, 0]], [np.radians(float(start_orientation))]])
        self.P_m = np.eye(3) # initial covariance matrix TODO: find some sense in initial covariance matrix

        # Process noise covariance matrix (Q)
        self.Q = np.diag([motor_noise_std**2, motor_noise_std**2, motor_noise_std**2])

        # Measurement noise covariance matrix (R)
        self.R = np.diag([gps_noise_std**2, gps_noise_std**2, imu_noise_std**2])

        self.u_x = float(0)
        self.u_theta = float(0)

        # Measurement matrix (H)
        self.H = np.array([[1, 0, 0], 
                           [0, 1, 0], 
                           [0, 0, 1]])
        
        
        # Control input model
        self.B = np.eye(3)

        self.motor_noise_std = motor_noise_std  # Standard deviation for motion noise
        self.gps_noise_std = gps_noise_std
        self.imu_noise_std = imu_noise_std

        self.z = self.true_x

        # Rotation PID parameters
        self.Kp_rot = 0.8
        self.Ki_rot = 0.5
        self.Kd_rot = 0.00
        self.previous_error_rot = 0.0
        self.integral_rot = 0.0
        self.previous_orientation = self.x_m[2, 0]
        self.integral_rot_max = 1.0
        self.integral_rot_min = - 1.0
        self.dt = 0.1

        # pos PID parameters
        self.Kp_pos = 0.5
        self.Ki_pos = 0.0
        self.Kd_pos = 0.005
        self.previous_error_pos = 0.0
        self.integral_pos = 0.0

    def calculate_distance_to_target(self, target_pos):
        return np.linalg.norm(target_pos - self.x_m)
    
    

    def calculate_estimated_angle_to_target(self, target_pos):
        """
        Calculates the angle from the current position to the target position.
        """
        delta_x = target_pos[0, 0] - self.x_m[0, 0]
        delta_y = target_pos[1, 0] - self.x_m[1, 0]
        angle_rad = np.arctan2(delta_y, delta_x)  # Angle in radians
        return angle_rad

    # PREDICTION USING KALMAN FILTER
    def predict(self):
        #self.move_step_pid_u(target_pos) TODO: Integrate the u directly in the moving function
        self.x_p = self.x_m + self.B @ np.array([[self.u_x*np.cos(self.x_m[2,0])],
                                                [self.u_x*np.sin(self.x_m[2,0])],
                                                [self.u_theta]])
        self.P_p = self.A @ self.P_m @ (self.A).T + self.Q
    
    # UPDATING KNOWING NEW INFORMATION
    def update(self):
        S = self.H @ self.P_p @ (self.H).T + self.R
        K = self.P_p @ (self.H).T @ np.linalg.inv(S)
        self.x_m = self.x_p + K @ (self.z - self.H @ self.x_p)
        self.P_m = self.P_p - self.P_p @ (self.H).T @ np.linalg.inv(S) @ self.H @ self.P_p

    def rotate_towards(self, target_pos):
        current_error = self.calculate_estimated_angle_to_target(target_pos) - self.x_m[2,0]
        self.integral_rot += current_error
        
        derivative = current_error - self.previous_error_rot
        rotation_amount = self.Kp_rot * current_error + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative
        self.x_m[2, 0] += rotation_amount
        self.x_m[2, 0] = self.x_m[2, 0] % 2*np.pi
        self.previous_error_rot = current_error
    
    def rot_step_pid_u(self, target_pos):
        current_error = self.calculate_estimated_angle_to_target(target_pos) - self.x_m[2,0]
        #print("Target_pos is", target_pos, "angle error is", current_error*180/3.14)
        #print('current_error is ', current_error)
        self.integral_rot += current_error
        self.integral_rot = max(min(self.integral_rot, self.integral_rot_max), self.integral_rot_min)

        # Calculate derivative based on measurement change
        current_orientation = self.x_m[2,0]
        derivative = (current_orientation - self.previous_orientation) / self.dt
        self.previous_orientation = current_orientation

        self.u_theta = self.Kp_rot * current_error + self.Ki_rot * self.integral_rot - self.Kd_rot * derivative
        

    def move_step_pid_u(self, target_pos):
        # Calculate the Euclidean distance to the target as the error
        current_error_pos = np.sqrt((self.x_m[0,0] - target_pos[0,0])**2 + (self.x_m[1,0] - target_pos[1,0])**2)

        # Update the integral of the error
        self.integral_pos += current_error_pos
        # Calculate the derivative of the error
        derivative_pos = current_error_pos - self.previous_error_pos
        # PID control for step size based on distance to target
        step_size = self.Kp_pos * current_error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        
        # Speed adjustment
        # You may need to adjust the max_speed and min_speed based on your robot's capabilities and testing
        max_speed = 5.0  # Maximum speed limit
        min_speed = 0.05  # Minimum speed to prevent too slow movements
        
        # Ensure the step size is within the min and max speed limits
        step_size = max(min(step_size, max_speed), min_speed)
        
        self.u_x = step_size
        self.previous_error_pos = current_error_pos

    def move_towards(self, target_pos):
        """
        Moves the robot towards the target position.
        """
        # PID control for step size based on distance to target
        current_error_pos = target_pos[0, 0] + target_pos[1, 0] - (self.position[0] + self.position[1])

        self.position += self.u_x * np.array([np.cos(self.orientation_rad), np.sin(self.orientation_rad)])
        self.previous_error_pos = current_error_pos

    def process_noise(self):
        return np.array([[np.random.normal(0, np.sqrt(self.Q[0,0]))],
                        [np.random.normal(0, np.sqrt(self.Q[1,1]))],
                        [np.random.normal(0, np.sqrt(self.Q[2,2]))]])
    
    def gps_measure(self):
        """
        Gives measured information on [x, y] position of the robot with measurement inaccuracy
        """
        #print(self.z)
        #print(self.z[:2, 0])
        #print(self.true_x[:2, 0])

        self.z[0:2] = self.true_x[0:2] + np.array([[np.random.normal(0,self.gps_noise_std)], 
                                                    [np.random.normal(0,self.gps_noise_std)]])
    

    def imu_sensor(self):
        """
        Simulates the sensing of the orientation with added noise.
        """
        self.z[2, 0] = self.true_x[2,0] + np.random.normal(0, self.imu_noise_std)
    
    def run(self, target_pos, steps=100, plot = False):
        estimated_positions = [self.x_m[:2, 0].copy()]  # Kalman filter estimates
        true_positions = [self.true_x[:2, 0].copy()]    # True positions
        gps_positions = [self.x_m[:2, 0].copy()] # Unfiltered estimates (for comparison)

        for k in range(steps):
            # Update control inputs using PID
            self.rot_step_pid_u(target_pos)
            self.move_step_pid_u(target_pos)

            # Apply process noise and move the robot
            process_noise = self.process_noise()
            self.true_x = self.A @ self.true_x + process_noise + self.B @ np.array([[self.u_x*np.cos(self.x_m[2,0])],
                                                                                    [self.u_x*np.sin(self.x_m[2,0])],
                                                                                    [self.u_theta]])
            # Simulate sensor measurements
            self.gps_measure()
            self.imu_sensor()

            # Perform Kalman filter prediction and update
            self.predict()
            self.update()

            # Store the positions
            estimated_positions.append(self.x_m[:2, 0].copy())
            true_positions.append(self.true_x[:2, 0].copy())
            gps_positions.append(self.z[:2, 0].copy())  # Assuming 'self.position' is the unfiltered estimate

            # Check for completion
            #print("Delta x", abs(self.x_m[0,0] - target_pos[0,0]))
            #print("Delta y", abs(self.x_m[1,0] - target_pos[1,0]))
            if np.sqrt((self.x_m[0,0] - target_pos[0,0])**2 + (self.x_m[1,0] - target_pos[1,0])**2) < 0.5:
                print(f"Reached target or very close to it in: {k} steps!")
                break
        
        if plot:
            #print(gps_positions)
            # Visualization
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(true_positions)[:, 0], np.array(true_positions)[:, 1], 'k-', label='True Position')
            plt.plot(np.array(estimated_positions)[:, 0], np.array(estimated_positions)[:, 1], 'b--', label='Kalman Filter Estimate')
            plt.plot(np.array(gps_positions)[:, 0], np.array(gps_positions)[:, 1], 'r:', label='Unfiltered Estimate')
            plt.plot(target_pos[0], target_pos[1], 'go', label='Target')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Robot Navigation with Kalman Filter')
            plt.legend()
            plt.grid(True)
            plt.show()

        return estimated_positions, true_positions, gps_positions



# Example usage
if __name__ == '__main__':
    start_pos = np.array([[8.5], 
                          [3.9]])
    robot = Robot(start_pos=start_pos, start_orientation=0, motor_noise_std=0.00002,
                  gps_noise_std=0.0005, imu_noise_std= 0.00005)
    target_pos = np.array([[17], 
                           [87]])
    steps = 200
    plotting = True
    positions, __, __ = robot.run(target_pos=target_pos, steps=steps, plot = plotting)

    # Set up the figure for animation again
    fig, ax = plt.subplots()
    ax.set_xlim(( -1, 100))
    ax.set_ylim((-1, 100))
    line, = ax.plot([], [], 'o-', lw=2)
    target, = ax.plot(target_pos[0, 0], target_pos[1, 0], 'rx', markersize=10)

    # Initialize animation
    def init():
        line.set_data([], [])
        target.set_data([target_pos[0]], [target_pos[1]])
        return (line, target)

    # Define animation function
    def animate(i):
        xdata = [pos[0] for pos in positions[:i+1]]
        ydata = [pos[1] for pos in positions[:i+1]]
        line.set_data(xdata, ydata)
        return (line,)

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(positions), interval=200, blit=False)
    plt.show()
