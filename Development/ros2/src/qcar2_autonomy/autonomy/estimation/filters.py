"""Kalman-filter primitives for QCar 2 state estimation.

These classes were originally embedded as private classes inside
``nav_to_pose.py``. They are extracted here so the upcoming ``ekf_fusor``
node and any other consumer (semantic mapper, active-SLAM monitor,
diagnostics) can reuse the same math without coupling to the path follower.

Math sources:
- Bicycle-model state propagation: QCar 2 User Manual - System Hardware v1.0.
- 2D EKF predict/correct: standard formulation (Thrun, Probabilistic Robotics).
- Wheelbase L: 0.256 m (manual Table 11).

    LC Commentary: Yes, maybe I went too far to strip EKF to here with bycicle motion model
        but I wanted to make test, and this is easier to test in isolation. 
        Also all of this is an pretty good start point architecture-wise for sensor fusion
            Prob used further and edited for Research At mdc with inclusion of camera vision
            sensor addition.
"""

import numpy as np

from pal.utilities.math import wrap_to_pi


class QcarEKF:
    """2D Extended Kalman Filter with a bicycle motion model.

    State: x = [x, y, theta]^T in some planar frame (typically ``map``).
    Input: u = [v, delta]  (linear speed, steering angle).
    Measurement: z = [x_meas, y_meas, theta_meas]  (e.g. Cartographer or AMCL pose).

    Wheelbase ``L`` defaults to 0.256 m per QCar 2 Table 11; override in
    constructor if a different platform is used.
    """

    def __init__(self, x0, P0, Q, R, L=0.256):
        self.L = L
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.C = np.eye(3)

    def f(self, X, u, dt):
        return X + dt * u[0] * np.array([
            [np.cos(X[2, 0])],
            [np.sin(X[2, 0])],
            [np.tan(u[1]) / self.L]
        ])

    def Jf(self, X, u, dt):
        return np.array([
            [1, 0, -dt * u[0] * np.sin(X[2, 0])],
            [0, 1,  dt * u[0] * np.cos(X[2, 0])],
            [0, 0, 1]
        ])

    def prediction(self, dt, u):
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        self.xHat = self.f(self.xHat, u, dt)
        self.xHat[2] = wrap_to_pi(self.xHat[2])

    def correction(self, y):
        H = self.C
        PH = self.P @ H.T
        S = H @ PH + self.R
        K = PH @ np.linalg.inv(S)
        z = y - H @ self.xHat
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)
        self.xHat += K @ z
        self.xHat[2] = wrap_to_pi(self.xHat[2])
        self.P = (self.I - K @ H) @ self.P


class GyroKF:
    """Two-state Kalman filter for yaw + yaw-rate-bias estimation.

    State: x = [theta, bias]^T.
    Input: u = gyro angular velocity (yaw rate).
    Measurement: z = absolute theta (e.g. from a global pose source).
    """

    def __init__(self, x0, P0, Q, R):
        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.A = np.array([[0, -1], [0, 0]])
        self.B = np.array([[1], [0]])
        self.C = np.array([[1, 0]])

    def prediction(self, dt, u):
        Ad = self.I + self.A * dt
        self.xHat = Ad @ self.xHat + dt * self.B * u
        self.P = Ad @ self.P @ Ad.T + self.Q

    def correction(self, y):
        PC = self.P @ self.C.T
        S = self.C @ PC + self.R
        K = PC @ np.linalg.inv(S)
        z = wrap_to_pi(y - self.C @ self.xHat)
        self.xHat += K @ z
        self.xHat[0] = wrap_to_pi(self.xHat[0])
        self.P = (self.I - K @ self.C) @ self.P
