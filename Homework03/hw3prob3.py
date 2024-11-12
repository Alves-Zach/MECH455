from math import atan
from turtle import color
from matplotlib import lines
from matplotlib.animation import adjusted_figsize
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib.patches import Ellipse

# Global variables
wGoal = np.array([0.5, 0.2, 0.3])
muGoal = np.array([[0.35, 0.38], [0.68, 0.25], [0.56, 0.64]])
EGoal = np.array([[[0.01, 0.004], [0.004, 0.01]],
              [[0.005, -0.003], [-0.003, 0.05]],
              [[0.008, 0.0], [0.0, 0.004]]])

def gmm_pdf(x, means, covariances, weights):
    """Calculate the probability density function of a GMM at a given point x."""
    pdf = np.zeros(x.shape[0])
    for mean, cov, weight in zip(means, covariances, weights):
        pdf += weight * multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
    return pdf

def generate_grid(x_min, x_max, y_min, y_max, n_points):
    """Generate a grid of points in 2D space."""
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.flatten(), yy.flatten()])
    return grid

# Generate the grid of points
grid = generate_grid(0, 1, 0, 1, 100)

# Generate the proabbilities of the GMM at each point in the grid
pdf = gmm_pdf(grid, muGoal, EGoal, wGoal).reshape(100, 100)

probabilitiesGrid = pdf.reshape(100, 100)

# The start of the EM algorithm
nSamples = 100
numClusters = 3
numIterations = 5

# Gamma matrix
gamma = np.zeros((nSamples, numClusters))

# Plot the probabilities in the center plots
fig, ax = plt.subplots(numIterations, 3)
plt.get_current_fig_manager().set_window_title('EM Algorithm')

# Giving titles to the first plots
ax[0, 0].set_title('Unlabed samples')
ax[0, 1].set_title('Ground truth')

# Generate samples based on the known weight mean cov
samples = np.clip(np.concatenate([np.random.multivariate_normal(muGoal[0], EGoal[0], 33),
                                  np.random.multivariate_normal(muGoal[1], EGoal[1], 33),
                                  np.random.multivariate_normal(muGoal[2], EGoal[2], 34)]),
                   0, 1)

# List of color associations for the samples
colors = ['red', 'blue', 'green']
colorArray = [None]*nSamples

# Initializing the guesses for the parameters
w = np.array([0.5, 0.15, 0.35])
mu = np.mean(samples, axis=0) + np.random.normal(0.0, 0.05, (numClusters, 2))
E = np.array([jnp.cov(samples.T) + np.random.normal(0.0, 0.005) for _ in range(numClusters)])

# While loop to wait until convergence criterion is met
iteration = 0
for n in range(numIterations):
    # E step
    for i in range(nSamples):
        # Loop to create the gamma matrix
        k = 0
        for weight, mean, cov in zip(w, mu, E):
            gamma[i, k] = weight * multivariate_normal.pdf(samples[i], mean=mean, cov=cov) /\
                np.sum([w[j] * multivariate_normal.pdf(samples[i], mean=mu[j], cov=E[j]) for j in range(numClusters)])

            k += 1

        # Assigning colors based on gamma values
        colorArray[i] = colors[np.argmax(gamma[i])]

    # M Step
    for k in range(numClusters):
        # Calculating sumations
        gammaSum = jnp.sum(gamma[:, k])
        gamma_xSum = jnp.append(jnp.sum(jnp.multiply(gamma[:, k], samples[:, 0])),
                                jnp.sum(jnp.multiply(gamma[:, k], samples[:, 1])))

        # Calculate the new mu
        mu[k] = jnp.divide(gamma_xSum, gammaSum)

        # Calculating sample - mu using new mu
        diff = samples - mu[k]  # Difference between each point and the mean of component k
        weighted_diff = np.expand_dims(gamma[:, k], axis=1) * diff  # Weighted difference

        # Calculate the new E
        E[k] = jnp.dot(weighted_diff.T, diff) / gammaSum

        # Calculate the new weights
        w[k] = jnp.divide(gammaSum, nSamples)

    # Visualizing the results
    # Plot the unlabeled samples on the left plot
    ax[iteration, 0].scatter(samples[:, 0], samples[:, 1], color='black', alpha=0.8)
    ax[iteration, 0].set_aspect('equal', 'box')
    ax[iteration, 0].set_xlim(0, 1)
    ax[iteration, 0].set_ylim(0, 1)

    # Plot the ground truth on the center plot
    ax[iteration, 1].scatter(samples[0:34, 0], samples[0:34, 1], color=colors[0], alpha=0.8)
    ax[iteration, 1].scatter(samples[34:67, 0], samples[34:67, 1], color=colors[1], alpha=0.8)
    ax[iteration, 1].scatter(samples[67:100, 0], samples[67:100, 1], color=colors[2], alpha=0.8)
    ax[iteration, 1].set_aspect('equal', adjustable='box')
    ax[iteration, 1].set_xlim(0, 1)
    ax[iteration, 1].set_ylim(0, 1)

    # Plot the points with thier associations on the right plot
    ax[iteration, 2].scatter(samples[:, 0], samples[:, 1], color=colorArray, alpha=0.8)
    ax[iteration, 2].set_xlim(0, 1)
    ax[iteration, 2].set_ylim(0, 1)
    ax[iteration, 2].set_aspect('equal', adjustable='box')

    # Getting the covarience to plot the elipse for each cluster
    for k in range(numClusters):
        # Adding the ellipses for the guesses
        eigVals, eigVectors = np.linalg.eig(E[k])
        patch1 = Ellipse(mu[k], width=3*np.sqrt(eigVals[0]), height=3*np.sqrt(eigVals[1]),
                         angle=np.degrees(np.arctan2(eigVectors[1, 0], eigVectors[0, 0])),
                         color=colors[k], fill=False, linewidth=2, linestyle='dashed')
        ax[iteration, 2].add_patch(patch1)

        # Adding the ellipses for the known parameters
        eigVals, eigVectors = np.linalg.eig(EGoal[k])
        patch2 = Ellipse(muGoal[k], width=3*np.sqrt(eigVals[0]), height=3*np.sqrt(eigVals[1]),
                         angle=np.degrees(np.arctan2(eigVectors[1, 0], eigVectors[0, 0])),
                         color=colors[k], fill=False, linewidth=2, linestyle='dashed')
        ax[iteration, 1].add_patch(patch2)

        # Adding the heatmap to the center plots
        ax[iteration, 1].imshow(probabilitiesGrid, extent=(0, 1, 0, 1), origin='lower',
                                cmap='binary', alpha=0.75)


    iteration += 1

plt.show()
