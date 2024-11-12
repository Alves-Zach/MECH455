import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import solve_bvp
from scipy.integrate import quad
from scipy.integrate import solve_ivp

###### Helper functions for the iLQR algorithm ######
# Starting parameters
x0 = np.array([0.3, 0.3, np.pi/2.0])
dt = 0.1
nSamples = 100

Qx = np.diag([10.0, 10.0, 2.0])
Ru = np.diag([4.0, 2.0])
P1 = np.diag([20.0, 20.0, 5.0])

Qz = np.diag([5.0, 5.0, 1.0])
Rv = np.diag([2.0, 1.0])

init_u_traj = np.tile(np.array([-0.5, 1.5]), reps=(100, 1))
init_u_1 = init_u_traj[:, 0][0]
init_u_2 = init_u_traj[:, 1][0]

# Update the state based on a packed trajectory
def step(xIn, uIn):
    xNew = xIn + dt * dyn(xIn, uIn)
    return xNew

# Creates a trajectory based on a constant control input
def simTraj(x0, uTraj):
    tsteps = uTraj.shape[0]
    xTraj = np.zeros((tsteps, 3))
    xt = x0.copy()
    for t in range(tsteps):
        xtNew = step(xt, uTraj[t])
        xTraj[t] = xtNew.copy()
        xt = xtNew.copy()
    return xTraj

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
    # The loss function is the sum of the cost over all time steps
    uRuTerm = 0
    FkxTerm = 0
    for t in range(nSamples):
        # The u(t) @ R @ u(t) term
        uRuTerm += u[t] @ Ru @ u[t]

        # The Fkx term
        FkxTerm += (x[t] - xd(t)) @ Qx @ (x[t] - xd(t))
        
    # The final term
    FkxTerm += (x[-1] - xd(nSamples * dt)) @ P1 @ (x[-1] - xd(nSamples * dt))

    return uRuTerm + FkxTerm

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
    xTraj = simTraj(x0, u_traj)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((nSamples, 3, 3))
    B_list = np.zeros((nSamples, 3, 2))
    a_list = np.zeros((nSamples, 3))
    b_list = np.zeros((nSamples, 2))
    for t_idx in range(nSamples):
        t = t_idx * dt
        A_list[t_idx] = getA(xTraj[t_idx], u_traj[t_idx])
        B_list[t_idx] = getB(xTraj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, xTraj[t_idx])
        b_list[t_idx] = dldu(u_traj[t_idx])

    xd_T = np.array([2.0 * (nSamples - 1) * dt / np.pi, 0.0, np.pi / 2.0])
    p1 = P1 @ (xd_T - x_traj[-1])

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
    tlist = np.arange(nSamples) * dt
    res = solve_bvp(zp_dyn_list, zp_bc, tlist, np.zeros((6, nSamples)), max_nodes=100)
    zp_traj = res.sol(tlist).T

    z_traj = zp_traj[:,:3]
    p_traj = zp_traj[:,3:]

    v_traj = np.zeros((nSamples, 2))
    for _i in range(nSamples):
        At = A_list[_i]
        Bt = B_list[_i]
        at = a_list[_i]
        bt = b_list[_i]

        zt = z_traj[_i]
        pt = p_traj[_i]

        vt = -np.linalg.inv(Rv) @ (Bt.T @ pt - bt)
        v_traj[_i] = vt

    return v_traj

######## Ergodic trajectory ########
# Current robot location
x = x0

# History of robot locations
xHistory = np.array([x0[:2]])
xHistory = np.append(xHistory, [x[:2]], axis=0)

# History of control inputs
uHistory = np.array([x0[:2]])

# Helper function to move the robot
def move(xdot):
    # Limit the max speed
    xdot[0] = np.clip(xdot[0], -1, 1)
    xdot[1] = np.clip(xdot[1], -1, 1)

    # Move the robot
    x[0] += xdot[0] * dt
    x[1] += xdot[1] * dt

    # Store the current location
    global xHistory, uHistory
    xHistory = np.append(xHistory, [x], axis=0)
    uHistory = np.append(uHistory, [xdot], axis=0)

    return x[:2]

######## Mixed model distribution ########
gridSize = 100

# Creating the mixed model distribution
w = np.array([0.5, 0.2, 0.3])
mu = np.array([[0.35, 0.38],
               [0.68, 0.25],
               [0.56, 0.64]])
E = np.array([[[0.01, 0.004],
               [0.004, 0.01]],
              [[0.005, -0.003],
               [-0.003, 0.005]],
              [[0.008, 0.0],
               [0.0, 0.004]]])

# Returns a pdf and a meshgrid based on the given parameters
def mixedModel():
    x = np.linspace(0, 1, gridSize)
    y = np.linspace(0, 1, gridSize)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.flatten(), yy.flatten()])
    pdf = np.zeros(grid.shape[0])
    for mean, cov, weight in zip(mu, E, w):
        pdf += weight * multivariate_normal.pdf(grid, mean=mean, cov=cov, allow_singular=True)
    return pdf, xx, yy

########### The actual ergodic trajectory ###########
# Make the pdf a callable function
def pdf(x):
    return sum([w[i] * multivariate_normal.pdf(x, mean=mu[i], cov=E[i], allow_singular=True) for i in range(len(w))])

# Perform the iLQR
init_u_traj = np.tile(np.array([-0.5, 1.5]), reps=(nSamples, 1))
u_traj = init_u_traj.copy()
x_traj_init = simTraj(x0, u_traj)

obj_val = []

for _ in range(10):
    x_traj = simTraj(x0, u_traj)

    v_traj = ilqr_iter(x0, u_traj)

    gamma = 1.0
    alpha = 1e-04
    beta = 0.5

    cost = loss(0, x_traj, u_traj)
    obj_val.append(cost)

    while True:
        u_traj_new = u_traj + gamma * v_traj
        x_traj_new = simTraj(x0, u_traj_new)

        cost_new = loss(0, x_traj_new, u_traj_new)
        expected_reduction = alpha * gamma * np.sum(v_traj * v_traj)
        
        if cost_new <= cost + expected_reduction:
            break

        gamma *= beta

    u_traj += gamma * v_traj

### We are going to use 10 coefficients per dimension --- so 100 index vectors in tota
num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T  # this is the set of all index vectors

# define a 1-by-1 2D search space
L_list = np.array([1.0, 1.0])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 100),
    np.linspace(0, L_list[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = 1.0 / 99
dy = 1.0 / 99  # the resolution of the grids

# Compute the coefficients
coefficients = np.zeros(ks.shape[0])  # number of coefficients matches the number of index vectors
for i, k_vec in enumerate(ks):
    # step 1: evaluate the fourier basis function over all the grid cells
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  # we use NumPy's broadcasting feature to simplify computation
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)  # normalization term
    fk_vals /= hk

    # step 2: evaluate the spatial probabilty density function over all the grid cells
    pdf_vals = pdf(grids)  # this can computed ahead of the time

    # step 3: approximate the integral through the Riemann sum for the coefficient
    phik = np.sum(fk_vals * pdf_vals) * dx * dy
    coefficients[i] = phik

# Helper function to calculate numerical gradient
def numerical_gradient(f, x, h=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.array(x, copy=True)
        x_h[i] += h
        grad[i] = np.min((f(x_h) - f(x)) / h)
    return grad

# Armijo step size rule function
def Armijo_step_size(zeta_i, descent_direction, objective_function, alpha=0.3, beta=0.8):
    gamma_i = 1.0
    while objective_function(zeta_i + gamma_i * descent_direction) > \
          objective_function(zeta_i) + alpha * gamma_i * np.dot(numerical_gradient(objective_function, zeta_i), descent_direction):
        gamma_i *= beta
    return gamma_i

# Example implementation of f_k
def f_k(t, x):
    return np.array([np.cos(t), np.sin(t)])  # Placeholder function

# Compute phi_k
def phi_k(t, x):
    phi = np.zeros(len(w))
    for k in range(len(w)):
        phi[k] = w[k] * multivariate_normal.pdf(x, mean=mu[k], cov=E[k])
    return phi

# Derivative of phi_k
def d_phi_k(t, x):
    d_phi = np.zeros((len(w), len(x)))
    for k in range(len(w)):
        diff = x - mu[k]
        inv_cov = np.linalg.inv(E[k])
        d_phi[k] = w[k] * multivariate_normal.pdf(x, mean=mu[k], cov=E[k]) * (-inv_cov @ diff)
    return d_phi

# D(zeta) function implementation
def D(zeta, x):
    def integrand(t):
        phi = phi_k(t, x)
        d_phi = d_phi_k(t, x)
        return np.sum([phi[k] * (d_phi[k] @ zeta) for k in range(len(w))])

    integral = np.array([integrand(t) for t in np.linspace(0, T, 100)]).mean(axis=0)
    return integral

# Differential equation system for solving z(t), v(t), P(t), r(t)
def differential_system(t, y, A, B, Q, R, b, f):
    n = A.shape[0]
    z, v, P, r = y[:n], y[n:2*n], y[2*n:2*n**2].reshape(n, n), y[2*n**2:]
    dz_dt = A @ z + B @ v
    dv_dt = np.linalg.solve(R(t), B.T @ P @ z - b(t))
    dP_dt = P @ A + A.T @ P - Q + P @ B @ np.linalg.inv(R(t)) @ B.T @ P
    dr_dt = -A.T @ r - P @ f(t, z)
    return np.concatenate([dz_dt, dv_dt, dP_dt.flatten(), dr_dt])

# Updated ergodic control algorithm function
def ergodic_control_algorithm(D, zeta_0, epsilon, Armijo_step_size, x_0, f, T, dt, A, B, Q, R, b):
    global x
    zeta_i = zeta_0
    u_i = np.zeros_like(zeta_0)
    x = x_0  # Initialize global x with x_0

    def f_vector(t, y):
        return f(t, y[:len(x_0)])

    i = 0
    while np.linalg.norm(D(zeta_i, x) * zeta_i) > epsilon:
        i += 1
        print(f"Iteration {i}")
        print(f"zeta_i: {zeta_i}")
        print(f"D(zeta_i): {D(zeta_i, x)}")

        # Calculate descent direction
        def objective_function(zeta):
            D_zeta = D(zeta, x)
            return np.dot(D_zeta, zeta) + 0.5 * np.dot(zeta, zeta)

        gradient = numerical_gradient(objective_function, zeta_i)
        descent_direction = -gradient
        print(f"Gradient: {gradient}")
        print(f"Descent Direction: {descent_direction}")

        # Update zeta_i using minimization approach
        n = len(zeta_0)
        y0 = np.concatenate([np.zeros(n), np.zeros(n), np.eye(n).flatten(), np.zeros(n)])
        sol = solve_ivp(differential_system, [0, T], y0, args=(A, B, Q, R, b, f_vector), t_eval=[T])
        yT = sol.y[:, -1]
        zT, vT, PT, rT = yT[:n], yT[n:2*n], yT[2*n:2*n**2].reshape(n, n), yT[2*n**2:]
        zeta_i = -np.linalg.solve(R(T), B.T @ PT @ zT - b(T))
        print(f"Updated zeta_i: {zeta_i}")

        # Choose step size gamma_i using Armijo rule or other method
        gamma_i = Armijo_step_size(zeta_i, descent_direction, objective_function)
        print(f"Step size gamma_i: {gamma_i}")

        # Update the control
        u_i = u_i + gamma_i * descent_direction
        print(f"Updated u_i: {u_i}")

        # Debugging exit condition
        if i > 10:  # Arbitrary large number to prevent infinite loop during debugging
            print("Breaking due to too many iterations.")
            break

    return u_i

# Initial parameters
w = np.array([0.3, 0.7])  # Example mixture weights
mu = np.array([[0.5, 0.5], [0.7, 0.3]])  # Example means of Gaussian components
E = np.array([[[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]]])  # Example covariance matrices of Gaussian components

zeta_0 = np.array([3.0, 3.0])  # Initial guess
epsilon = 1e-6  # Convergence threshold
x_0 = np.array([0.0, 0.0])  # Initial state
f = lambda t, x: np.array([np.cos(t), np.sin(t)])  # Example system dynamics function

T = 1.0  # Time horizon
dt = 0.1  # Time step

# Define the A, B, Q, R matrices
A = np.array([[0, 1], [-1, 0]])  # Example dynamics matrix
B = np.eye(2)  # Example control matrix
Q = np.eye(2)  # Example state cost matrix
R = lambda t: np.eye(2)  # Example control cost matrix as a function of time
b = lambda t: np.array([0, 0])  # Example function b(t)

# History of control inputs
u_i_history = np.zeros((100, 2))

# Run the ergodic control algorithm 100 times and store the controls
for i in range(100):
    u_final = ergodic_control_algorithm(D, zeta_0, epsilon, Armijo_step_size, x_0, f, T, dt, A, B, Q, R, b)
    u_i_history[i] = u_final

    # Move the robot using the final u_i
    move(u_final)

# print("Control history u_i_history:", u_i_history)

######## Plotting ########
### Plot 1
# Plotting the mixed model distribution
pdf, xx, yy = mixedModel()
fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(15, 5))

# Grid marks for debugging
# ax.set_xticks(np.linspace(0, 1, gridSize + 1))
# ax.set_yticks(np.linspace(0, 1, gridSize + 1))
# ax.grid(which='both', color='black', linestyle='-', linewidth=1, alpha=1.0)
# ax.imshow(pdf.reshape(gridSize, gridSize), extent=(0, 1, 0, 1),
#           cmap='Oranges', origin='lower')

# Showing the history and current location of the robot[0][0]
ax[0].set_title('Robot movement')
ax[0].contourf(pdf.reshape(gridSize, gridSize), extent=(0, 1, 0, 1), cmap='Oranges', origin='lower', levels=8)
ax[0].plot(xHistory[:, 0], xHistory[:, 1], 'k-',
        zorder=1, marker="^", markersize=5)
ax[0].plot(x[0], x[1], 'bo', zorder=1, markersize=10)

# Setting other visual aspects of the plot
ax[0].set_aspect('equal')
ax[0].set_xlim(-1/gridSize, 1 + 1/gridSize)
ax[0].set_ylim(-1/gridSize, 1 + 1/gridSize)

### Plot 2
# Plotting the uHistory
ax[1].set_title('Control Input')
ax[1].plot(np.linspace(0, nSamples-1, nSamples-1), u_i_history[1:, 0], 'b-')
ax[1].plot(np.linspace(0, nSamples-1, nSamples-1), u_i_history[1:, 1], 'r-')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Control Input')
ax[1].legend(['u1', 'u2'])
ax[1].figure.set_size_inches(15, 5)

### Plot 3
# Plotting the objective value
ax[2].set_title('Objective Value')
ax[2].plot(np.linspace(0, 10, 10), obj_val, 'b-')
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Objective Value')
ax[2].legend(['Objective Value'])
ax[2].figure.set_size_inches(15, 5)

plt.show()
