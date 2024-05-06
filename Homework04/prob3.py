import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp

# Scipt parameters
dt = 0.1
tf = 2 * math.pi
x0 = np.array([0, 0, math.pi / 2])
N = int(tf / dt)
init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(N, 1))

# State
x = x0

# Desired trajectory
x_t = np.linspace(0, tf, N) * 4 * (1 / tf)
y_t = np.zeros(N)
theta_t = np.ones(N) * math.pi / 2

# Arrays for the optimal controls at each time step
u = np.zeros((N, 2))

# Array for the objective value at each time step
J = np.zeros(N)

##### Helper functions #####
# Update the state based on a packed trajectory
def step(trajIn):
    # Unpack the trajectory
    xdot = trajIn[0] * math.cos(robotState[2])
    ydot = trajIn[0] * math.sin(robotState[2])
    thetadot = trajIn[1]
    
    # Update the state based on the differential drive kinematics
    robotState[0] += xdot
    robotState[1] += ydot
    robotState[2] += thetadot

# Update the state based on a non-packed trajectory
def step2(v, w):
    # Update the state based on the differential drive kinematics
    robotState[0] += v * math.cos(robotState[2])
    robotState[1] += v * math.sin(robotState[2])
    robotState[2] += w

# Setting up figures and plots
fig, ax = plt.subplots(3, 1)
ax[0].set_title('State Trajectory', weight='bold')
ax[1].set_title('Optimal Control', weight='bold')
ax[2].set_title('Objective Value', weight='bold')

# Plotting the desired trajectory
ax[0].plot(x_t, y_t, label='Desired Trajectory')

# Making the plots look pretty
ax[0].set_xlim([-0.25, 4.25])
ax[0].set_ylim([-0.25, 2.25])
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')

ax[1].set_xlabel('Time')
ax[1].set_ylabel('Control')

# Show the plots
plt.show()
