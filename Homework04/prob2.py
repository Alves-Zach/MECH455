import math
from cv2 import line
import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 10e-3
beta = 0.5
N = 100

# Initial guess
x = np.array([-4, -2])

# Defining f(x)
def f(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - (0.46 * x[0] * x[1])

# The Armijo line search algorithm
def armijo(x, z):
    # Parameters
    global alpha, beta

    # Initial step size
    stepSize = 1

    # The loop to check if the Armijo condition is satisfied
    while f(x + stepSize * z) > f(x) + alpha * stepSize * np.dot(-np.transpose(z), z):
        stepSize *= beta

    # Return once the Armijo condition is satisfied
    return x + stepSize * z

# The main function
def main():
    global N, x

    # Storing the guesses to be plotted later
    xinit = x

    # The loop to find the minimum
    for i in range(N):
        # Update the descent direction
        z = -1 * nd.Gradient(f)(x)

        # Update the current guess
        x = armijo(x, z)

        # Store the current guess to be plotted later
        xinit = np.vstack((xinit, x))

    # Return the final guess
    return xinit

if __name__ == '__main__':
    # Run the main function
    xinit = main()

    # Plot the function
    x1 = np.linspace(-6, 6, 100)
    x2 = np.linspace(-6, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f([X1, X2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Armijo Line Search', weight='bold')
    ax.plot_surface(X1, X2, Z, cmap='Blues_r')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    ax.plot(xinit[:, 0], xinit[:, 1], f(xinit.T),
            color='r', zorder=10, alpha=0.75, linewidth=4)
    plt.show()