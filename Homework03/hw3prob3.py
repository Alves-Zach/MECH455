import math
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Global variables
w = np.array([0.5, 0.2, 0.3])
mu = np.array([0.35, 0.38], [0.68, 0.25], [0.56, 0.64])
E1 = np.array([0.01, 0.004], [0.004, 0.01])
E2 = np.array([0.005, -0.003], [-0.003, 0.05])
E3 = np.array([0.008, 0.0], [0.0, 0.004])

# Generate a list of samples
