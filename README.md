# Kalman filtering of GPS and IMU data for vehicle motion planning
Ongoing personal hobby project using Kalman filters and PID controllers, with different graph search techniques to lead an autonomous vehicle on a randomised environment. Working on graph search implementation and obstacle avoidance at the moment.

![image](https://github.com/raphaellevisse/kalman_pid_estimation/assets/143650581/8c10309b-e77f-403d-aba6-9850b5296d62)
Just launch main.py and you can adjust the different parameters influencing movement. We see the use of Kalman filtering here as gps information in red is noisy, we rely on process noise and our knowledge of the vehicle equations to derive a reliable robot.

In actual version, I simulate a randomised factory floor, on which the robot uses A* star algorithm to go to a point given by the user by clicking on the grid.
![full_photo](https://github.com/raphaellevisse/kalman_pid_estimation/assets/143650581/6c455efe-2467-453c-9dcb-61b829fca217)


