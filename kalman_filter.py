import numpy as np


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, A, H, Q, R):
        self.state = initial_state
        self.covariance = initial_covariance
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self):
        self.state = self.A @ self.state
        self.covariance = self.A @ self.covariance @ self.A.T + self.Q

    def update(self, measurement):
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state += K @ (measurement - self.H @ self.state)
        self.covariance -= K @ self.H @ self.covariance
