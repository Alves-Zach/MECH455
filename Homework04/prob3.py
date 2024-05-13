from operator import inv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_bvp
from matplotlib.legend_handler import HandlerTuple

# iLQR parameters
dt = 0.1
tf = 2 * math.pi
x0 = np.array([0, 0, math.pi / 2.0])
N = 63
init_u_traj = np.tile(np.array([-0.5, 1.5]), reps=(N, 1))
termCondition = np.array([4, 0, np.pi / 2.0])

# State
x = x0

# Desired trajectory
x_d = np.array([[2.0 * x / np.pi, 0.0] for x in np.arange(0, 6.3, 0.1)])

##### Helper functions #####
# Update the state based on a packed trajectory
def step(xIn, uIn):
    xNew = xIn + dt * dyn(xIn, uIn)
    return xNew

# Creates a trajectory based on a constant control input
def createInitTraj(startingPoint, u_traj):
    tsteps = u_traj.shape[0]
    xTraj = np.zeros((tsteps, 3))
    xt = startingPoint.copy()
    for t in range(tsteps):
        xt_new = step(xt, u_traj[t])
        xTraj[t] = xt_new.copy()
        xt = xt_new.copy()
    return xTraj

################## Starting the iLQR algorithm

# Qx, Ru, P1, Qz, Rv
Qx = np.diag([10.0, 10.0, 2.0])
Ru = np.diag([4.0, 2.0])
P1 = np.diag([20.0, 20.0, 5.0])
Qz = np.diag([5.0, 5.0, 1.0])
Rv = np.diag([2.0, 1.0])

# Initial control trajectory
uInit1 = init_u_traj[:, 0][0]
uInit2 = init_u_traj[:, 1][0]

##### Helper functions to get the terms for the iLQR algorithm #####
# Get the xdot term
def dyn(xIn, uIn):
    return np.array([uIn[0] * math.cos(xIn[2]),
                     uIn[0] * math.sin(xIn[2]),
                     uIn[1]])

# Get the A matrix
def getA(xIn, uIn):
    # The A matrix is the partial derivative of dynamics wrt x
    return np.array([[0, 0, -uIn[0] * math.sin(xIn[2])],
                     [0, 0, uIn[0] * math.cos(xIn[2])],
                     [0, 0, 0]])

# Get the B matrix
def getB(xIn, uIn):
    # The B matrix is the partial derivative of dynamics wrt u
    return np.array([[math.cos(xIn[2]), 0],
                     [math.sin(xIn[2]), 0],
                     [0, 1.0]])

# Get the loss function
def loss(t, x, u):
    totalLoss = 0
    for t in range(N):
        xd = np.array(
            [2.0 * t * dt / np.pi, 0.0, np.pi / 2.0]
        )  # update desired state at each timestep
        xLoss = (x[t] - xd).T @ Qx @ (x[t] - xd)
        uLoss = u[t].T @ Ru @ u[t]
        totalLoss += xLoss + uLoss
    return totalLoss

# Get x desired at time t
def xd(t):
    return np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])

# Get dldx
def dldx(t, x):
    # Take the derivative of l(x, u) wrt x
    return 2 * Qx @ (x - xd(t))

# Get dldu
def dldu(u):
    # Take the derivative of l(x, u) wrt u
    return 2 * Ru @ u

###### iLQR Algorithm ######
def ilqr_iter(x0, u_traj):
    # :param x0: initial state of the system
    # :param u_traj: current estimation of the optimal control trajectory
    # :return: the descent direction for the control
    # forward simulate the state trajectory
    xTraj = createInitTraj(x0, u_traj)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((N, 3, 3))
    B_list = np.zeros((N, 3, 2))
    a_list = np.zeros((N, 3))
    b_list = np.zeros((N, 2))
    for t_idx in range(N):
        t = t_idx * dt
        A_list[t_idx] = getA(xTraj[t_idx], u_traj[t_idx])
        B_list[t_idx] = getB(xTraj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, xTraj[t_idx])
        b_list[t_idx] = dldu(u_traj[t_idx])

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
        m_1 = Bt @ np.linalg.inv(Rv) @ (-Bt.T @ zp[3:] - bt)
        m_2 = -at - At.T @ zp[3:]
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
        return np.hstack([zp_0[:3], zp_T[:3] - p1])

    ### The solver will say it does not converge, but the returned result
    ### is numerically accurate enough for our use
    tlist = np.arange(N) * dt
    res = solve_bvp(zp_dyn_list, zp_bc, tlist, np.zeros((6,N)), max_nodes=100)
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

        vt = -np.linalg.inv(Rv) @ (Bt.T @ pt - bt)
        v_traj[_i] = vt

    return v_traj

# Starting conditions for the iLQR algorithm
uTraj = init_u_traj.copy()
initXTraj = createInitTraj(x0, uTraj)

# Array to store the objective value
objVal = []

# The loop to iterate through the iLQR algorithm
for iter in range(10):
    # Simulate with the current command
    xTraj = createInitTraj(x0, init_u_traj)

    decent = ilqr_iter(x0, uTraj)

    gamma = 1.0
    alpha = 1e-04
    beta = 0.5

    cost = loss(0, xTraj, uTraj)
    objVal.append(cost)

    while True:
        uTrajNew = uTraj + gamma * decent
        xTrajNew = createInitTraj(x0, uTrajNew)

        cost_new = loss(0, xTrajNew, uTrajNew)
        expected_reduction = alpha * gamma * np.sum(decent * decent)
        print(expected_reduction, cost, cost_new)

        if cost_new <= cost + expected_reduction:
            break

        gamma *= beta

    uTraj += gamma * decent

# Setting up figures and plots
fig, ax = plt.subplots(1, 3)
ax[0].set_title('State Trajectory', weight='bold')
ax[1].set_title('Optimal Control', weight='bold')
ax[2].set_title('Objective Value', weight='bold')

final_line = ax[0].plot(xTraj[:, 0], xTraj[:, 1], linestyle="-", color="k")
init_line = ax[0].plot(initXTraj[:, 0], initXTraj[:, 1], linestyle="--", color="k")
desired = ax[0].plot(x_d[:, 0], x_d[:, 1], linestyle="-", color="r")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend(
    [(init_line[0],), (final_line[0],), (desired[0])],
    ["Initial Trajectory", "Converged Trajectory", "Desired Trajectory"],
    handler_map={(init_line[0], final_line[0], desired[0]): HandlerTuple(ndivide=None)},
)
ax[0].set_ylim(-2, 2)
ax[1].set_title("Optimal Control")
u1_line = ax[1].plot(np.arange(0, 6.3, 0.1), uTraj[:, 0], linestyle="-", color="b")
u2_line = ax[1].plot(np.arange(0, 6.3, 0.1), uTraj[:, 1], linestyle="-", color="r")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Control")
ax[1].legend(
    [(u1_line[0],), (u2_line[0],)],
    ["u_1(t)", "u_2(t)"],
    handler_map={(u1_line[0], u2_line[0]): HandlerTuple(ndivide=None)},
)
ax[2].set_title(
    "Objective Function init u:(u1={}, u2 ={})".format(uInit1, uInit2)
)
ax[2].plot(objVal, linestyle="-", color="b")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Objective")
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)

ax[2].set_box_aspect(1)

# Show the plots
plt.show()
