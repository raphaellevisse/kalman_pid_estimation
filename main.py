from environment import Environment
from robot import Robot
from kalman_filter import KalmanFilter
import numpy as np

import matplotlib.pyplot as plt

# Define constants
MOTOR_NOISE_STD = 0.1
GPS_NOISE_STD = 10.0
IMU_NOISE_STD = 1.0

FACTORY_WIDTH = 100
FACTORY_HEIGHT = 100
START_POINT = np.array([[0], [0]])

INITIAL_STATE = START_POINT
INITIAL_COVARIANCE = np.array([[1, 0],[0, 1]])


A = np.array(None)
H = np.array(None)
Q = np.array(None)
R = np.array(None)

# Instantiate classes
robot = Robot(START_POINT, MOTOR_NOISE_STD, GPS_NOISE_STD, IMU_NOISE_STD)
environment = Environment(FACTORY_WIDTH, FACTORY_HEIGHT)
kalman_filter = KalmanFilter(START_POINT, INITIAL_COVARIANCE, A, H, Q, R)
print("Target is", environment.target)


# Simulation loop
k = 0
N = 100
target_positions = np.zeros((2, N))



