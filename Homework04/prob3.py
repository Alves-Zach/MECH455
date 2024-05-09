import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp

# iLQR parameters
dt = 0.1
tf = 2 * math.pi
x0 = np.array([0, 0, math.pi / 2])
N = math.ceil(tf / dt) + 1
init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(N, 1))
termCondition = np.array([4, 0, np.pi / 2.0])

# State
x = x0

# Desired trajectory
x_d = np.array([4 * np.pi / 2.0, 0.0, -np.pi / 2.0])

##### Helper functions #####
# Update the state based on a packed trajectory
def step(trajIn):
    step2(trajIn[0], trajIn[1])

# Update the state based on a non-packed trajectory
def step2(v, w):
    # Update the state based on the differential drive kinematics
    x[0] += v * math.cos(x[2])
    x[1] += v * math.sin(x[2])
    x[2] += w

# Step function that does not update the global state and u has 3 elements
def step3(x, u):
    # Grabbing the current state
    output = np.copy(x)

    # Update the state based on the differential drive kinematics
    output[0] += u[0] * math.cos(x[2])
    output[1] += u[0] * math.sin(x[2])
    output[2] += u[1]

    return output

# Creates a trajectory based on a constant control input
def createInitTraj(startingPoint = x0, u_traj = init_u_traj):
    # Create the trajectory array
    traj = np.zeros((N, 3))

    # Set the starting point
    traj[0] = startingPoint

    # Create the trajectory
    for i in range(1, N):
        traj[i] = step3(traj[i - 1], u_traj[i] * dt)

    return traj

# Array to store the state trajectory
xTraj = np.zeros((N, 3))

# Array to store the control trajectory
uTraj = np.zeros((N, 2))

################## Starting the iLQR algorithm

# Qx, Ru, P1, Qz, Rv
Qx = np.diag([10.0, 10.0, 2.0])
Ru = np.diag([4.0, 2.0])
P1 = np.diag([20.0, 20.0, 5.0])
Qz = np.diag([5.0, 5.0, 1.0])
Rv = np.diag([2.0, 1.0])

##### Helper functions to get the terms for the iLQR algorithm #####
# Get the xdot term
def get_xdot(x, u):
    return np.array([u[0] * math.cos(x[2]), u[0] * math.sin(x[2]), u[1]]) * dt

# Get the A matrix
def getA(x, u):
    # The A matrix is the partial derivative of dynamics wrt x
    return np.array([[1, 0, -u[0] * math.sin(x[2]) * dt],
                     [0, 1, u[0] * math.cos(x[2]) * dt],
                     [0, 0, 1]])

# Get the B matrix
def getB(x, u):
    # The B matrix is the partial derivative of dynamics wrt u
    return np.array([[math.cos(x[2]) * dt, 0],
                     [math.sin(x[2]) * dt, 0],
                     [0, dt]])

# Get the loss function
def loss(t, x, u):
    # The current position in the desired trajectory
    xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])

    # The loss for x and u
    xLoss = (x - xd).T @ Qx @ (x - xd)
    uLoss = u.T @ Ru @ u

    return xLoss + uLoss

# Get x desired at time t
def xd(t):
    return np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])

# Get dldx
def dldx(t, x):
    # Take the derivative of l(x, u) wrt x
    return Qx @ (x - xd(t)) + Qx.T @ (x - xd(t))

# Get dldu
def dldu(u):
    # Take the derivative of l(x, u) wrt u
    return Ru @ u + Ru.T @ u

###### iLQR Algorithm ######
def ilqr_iter(x0, u_traj):
    # :param x0: initial state of the system
    # :param u_traj: current estimation of the optimal control trajectory
    # :return: the descent direction for the control
    # forward simulate the state trajectory
    x_traj = createInitTraj(x0, u_traj)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((N, 3, 3))
    B_list = np.zeros((N, 3, 2))
    a_list = np.zeros((N, 3))
    b_list = np.zeros((N, 2))
    for t_idx in range(N):
        t = t_idx * dt
        A_list[t_idx] = getA(x_traj[t_idx], u_traj[t_idx])
        B_list[t_idx] = getB(x_traj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(x_traj[t_idx], u_traj[t_idx])
        b_list[t_idx] = dldu(x_traj[t_idx], u_traj[t_idx])

    xd_T = np.array([
        2.0*(N-1)*dt / np.pi, 0.0, np.pi/2.0
    ])  # desired terminal state

    p1 = termCondition

    def zp_dyn(t, zp):
        # Getting the A and B matrices
        t_idx = (t/dt).astype(int)
        At = A_list[t_idx]
        Bt = B_list[t_idx]
        at = a_list[t_idx]
        bt = b_list[t_idx]

        # M matrix
        M_11 = At
        M_12 = np.zeros((3,3))
        M_21 = np.zeros((3,3))
        M_22 = -At.T

        # Dynamic matrix
        dyn_mat = np.block([
            [M_11, M_12],
            [M_21, M_22]
        ])

        # Dynamic vector
        m_1 = Bt * (-bt - pt @ Bt) * Rv
        m_2 = zt.T @ Qx + at
        dyn_vec = np.hstack([m_1, m_2])

        return dyn_mat @ zp + dyn_vec

    # this will be the actual dynamics function you provide to solve_bvp,
    # it takes in a list of time steps and corresponding [z(t), p(t)]
    # and returns a list of [zdot(t), pdot(t)]
    def zp_dyn_list(t_list, zp_list):
        list_len = len(t_list)
        zp_dot_list = np.zeros((6, list_len))
        for _i in range(list_len):
            zp_dot_list[:,_i] = zp_dyn(t_list[_i], zp_list[:,_i])
        return zp_dot_list

    # boundary condition (inputs are [z(0),p(0)] and [z(T),p(T)])
    def zp_bc(zp_0, zp_T):
        z0 = np.array([0, 0, np.pi/2])

    ### The solver will say it does not converge, but the returned result
    ### is numerically accurate enough for our use
    # zp_traj = np.zeros((N,6))  # replace this by using solve_bvp
    tlist = np.arange(N) * dt
    res = solve_bvp(
        zp_dyn_list, zp_bc, tlist, np.zeros((6,N)),
        max_nodes=100
    )
    zp_traj = res.sol(tlist).T

    z_traj = zp_traj[:,:3]
    p_traj = zp_traj[:,3:]

    v_traj = np.zeros((N, 2))
    for _i in range(N):
        At = A_list[_i]
        Bt = B_list[_i]
        at = a_list[_i]
        bt = b_list[_i]

        zt = z_traj[_i]
        pt = p_traj[_i]

        vt = np.zeros(2)  # replace this
        v_traj[_i] = vt

    return v_traj

# Setting up figures and plots
fig, ax = plt.subplots(1, 3)
ax[0].set_title('State Trajectory', weight='bold')
ax[1].set_title('Optimal Control', weight='bold')
ax[2].set_title('Objective Value', weight='bold')

# Plotting the desired trajectory
initTraj = createInitTraj()
ax[0].plot(initTraj[:, 0], initTraj[:, 1], label='Initial Trajectory', color='black', linestyle='--')
ax[0].legend()

# Making the plots look pretty
ax[0].set_xlim([-0.25, 4.25])
ax[0].set_ylim([-0.25, 2.25])
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_aspect('equal')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Control')

# Showing the iterations
fig2, ax2 = plt.subplots(1, 10)

# Setting the current command to start at the initial command
curCommand = init_u_traj

# For loop to iterate through the iLQR algorithm
for iter in range(10):
    # Simulate with the current command
    xTraj = createInitTraj(u_traj=curCommand)

    # Visualize the current trajectory
    ax2[0].plot(xTraj[:, 0], xTraj[:, 1], label='Iteration ' + str(iter))
    ax2[0].set_title('State Trajectory')
    ax2[0].set_xlim([-0.25, 4.25])
    ax2[0].set_ylim([-0.25, 2.25])
    ax2[0].set_aspect('equal')
    ax2[0].legend()

    # Get descent direction
    descent = ilqr_iter(x0, curCommand)

    # Line search params
    gamma = 1.0
    alpha = 1e-4
    beta = 0.5

    # Update step sizes

    # Update the command
    curCommand += gamma * descent

# Show the plots
plt.show()
