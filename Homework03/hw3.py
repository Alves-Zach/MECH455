import math
from matplotlib import markers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import jax.numpy as jnp
from pathlib import Path

def prob1Setup():
    # Global variables
    global image, xcoords, ycoords, dx, dy, samplePoints, readings, sampleWeights

    # Image file
    imagePath = Path(__file__).resolve().parent / 'lincoln.jpg'
    image = jnp.subtract(255, mpimg.imread(imagePath))

    # Getting x and y coordinates
    xcoords = np.linspace(0.0, 1.0, image.shape[0])
    ycoords = np.linspace(0.0, 1.0, image.shape[1])

    dx = xcoords[1] - xcoords[0]
    dy = ycoords[1] - ycoords[0]

    # Normalizing the image grid
    image = jnp.divide(image, jnp.sum(image) * dx * dy)

    # List of points sampled and their corresponding readings
    samplePoints = np.empty((0, 2))
    readings = np.empty((0, 1))
    sampleWeights = np.empty((0, 2))

###### Helper functions ######
# Get the chances of a positive reading at the given point
def getChance(point):
    # Get the darkness of the point
    posChances = image[math.floor(point[0] * image.shape[0]),
                       math.floor(point[1] * image.shape[1])]

    return posChances

# Get the reading at a given point
def getReading(point):
    # Get the chances of a positive reading at a given point
    posChances = getChance(point)

    # Get a random number between 0 and 1
    randNum = np.random.rand()

    # If the random number is less than the chances of a positive reading, then the reading is positive
    if randNum < posChances:
        return 1
    else:
        return 0

# Get the samples from the image
def getSamples(numSamples):
    # Get global variables
    global samplePoints, readings, sampleWeights
    
    # Generate a list of samples
    samplePoints = np.random.uniform(low=0.0, high=1.0, size=(numSamples, 2))
    
    # Setting up the sampleWeights array
    sampleWeights = np.zeros(numSamples)

    # Loop over all of the sample points
    for i in range(0, numSamples):
        # Get the weight from the image
        sampleWeights[i] = getDensity(samplePoints[i])

    # Normalize the sample weights
    sampleWeights /= np.max(sampleWeights)

# Get image density from the image
def getDensity(point):
    # Getting global variables
    global xcoords, ycoords

    # Find the closest pixel to the point requested
    pixel = np.array([np.argmin(np.abs(xcoords - point[0])),
                      np.argmin(np.abs(ycoords - point[1]))])
    
    # Getting the value of the image at the pixel point
    return image[pixel[0], pixel[1]]

# Main function
def prob1():
    # Setup
    prob1Setup()

    # Getting global variables
    global samplePoints, readings

    # Create a mutliplot, one for the image and one for the samples
    fig, ax = plt.subplots(1, 2)

    # Getting x number of samples on the image based on the darkness of the image
    numSamples = 5000
    getSamples(numSamples)

    # Display the image
    ax[0].set_title('Image')
    ax[0].imshow(jnp.subtract(255, image), cmap='gray', extent=[0, 1, 0, 1])
    ax[0].set_aspect('equal')

    # Display the weights plot
    ax[1].set_title('Weights')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_aspect('equal')
    
    # Displaying the points with their alphas
    global sampleWeights
    for point, weight in zip(samplePoints, sampleWeights):
        ax[1].plot(point[1], -point[0] + 1, linestyle='', marker='o', markersize=4, color='k', alpha=weight)

    # # Formating the figure
    fig.set_label('Image and Samples')
    fig.canvas.manager.set_window_title('Image and Samples')
    plt.show()

###### Problem 2 global variables ######

# Time and timestep
tf = 6.3
dt = 0.1

# Number of particles
numParticles = 100

# The state at time = 0
groundTruth = np.array([0.0, 0.0, np.pi/2])
measuredState = np.array([0.0, 0.0, np.pi/2])

# List of particles and their weights
xbar = np.zeros(4)

# The constant control signal
controlSignal = np.array([1.0, -0.5])

# Sensor noise
sensorNoise = 0.02

# State of the robot from t = 0
stateHistory = np.empty((0, 3))

# List of particles
particles = np.zeros((numParticles, 3))
particles[:, 0] = np.random.normal(0.0, 0.2, numParticles)
particles[:, 1] = np.random.normal(0.0, 0.2, numParticles)
particles[:, 2] = np.random.rand(numParticles) * 2 * np.pi
particles[:, 2] += np.pi / 2

# Chosen particles
chosenParticles = np.zeros((63, 3))

###### Problem 2 helper functions ######
# Get the state based on an input
def getStateUpdate(u1, u2):
    # Get global variables
    global groundTruth
    
    # Getting the change in state
    dx = u1 * jnp.cos(groundTruth[2])
    dy = u1 * jnp.sin(groundTruth[2])
    dtheta = u2

    return jnp.array([dx, dy, dtheta])

# Generate new particles based on the current state
def newParticles():
    # Get global variables
    global particles, numParticles

    # Generate new particles
    particles[:, 0] = np.random.normal(groundTruth[0], 0.1, numParticles)
    particles[:, 1] = np.random.normal(groundTruth[1], 0.1, numParticles)
    particles[:, 2] = np.random.normal(groundTruth[2], 0.1, numParticles)
    particles[:, 2] += np.pi / 2 # To account for the starting angle

# Get measurement based on command signal
def getMeasurement(u1, u2):
    # Get global variables
    global groundTruth, sensorNoise, measuredState
    
    # Estimate the measurement based on the ground truth
    measuredState = getStateUpdate(u1, u2) + np.random.normal(0, sensorNoise, 3)

# Calculate the weights of the particles
def calcWeights():
    # Get global variables
    global particles, measuredState, sensorNoise, weights

    # Calculate the weights based on the distances from the measured state
    particleDistances = np.linalg.norm(particles[:, :2] - measuredState[:2], axis=1)

    # Calculate the particle probabilities
    probabilities = np.exp(np.multiply(jnp.square(particleDistances), -0.5 * sensorNoise))
    
    # Calc weights
    weights = probabilities / np.sum(probabilities)

# Resample the particles based on weights
def resampleParticles():
    # Get global variables
    global particles, weights, numParticles, chosenParticles

    # Get the indices of the particles
    indices = np.random.choice(range(numParticles), size=numParticles, p=weights)

    # Resample the particles
    chosenParticles = particles[indices]

# Update the state based on input
# By default the time step is 0.1
def updateState(u1, u2):
    # Get global variables
    global groundTruth, stateHistory

    # Update the state
    stateUpdate = getStateUpdate(u1/10, u2/10)
    groundTruth[0] += stateUpdate[0]
    groundTruth[1] += stateUpdate[1]
    groundTruth[2] += stateUpdate[2]

    # Store the state of the robot in the global variable
    stateHistory = np.vstack((stateHistory, groundTruth))

# Problem 2
def prob2():
    # Getting global variables
    global chosenParticles, controlSignal
    
    # List for particle colors
    colors = ['r', 'y', 'g', 'b', 'c', 'm', 'k', 'w']
    curColor = 0

    # Run the state update for 6 seconds
    for i in range(0, 63):
        # Genrate new particles based on the current state
        newParticles()
        # Update ground truth
        updateState(controlSignal[0], controlSignal[1])
        
        # Get a measurement of the state
        getMeasurement(controlSignal[0], controlSignal[1])

        # Calculate the weights of the particles
        calcWeights()

        # Resample the particles
        resampleParticles()

        # Plot the particles at specific intervals
        if i % 3 == 0:
            plt.scatter(particles[:, 0], particles[:, 1], color=colors[curColor],
                        edgecolors='k')
        if i % 9 == 0:
            curColor += 1

    # Plot the particles at the last position
    plt.scatter(particles[:, 0], particles[:, 1], color=colors[curColor],
                edgecolors='k')
    


    # Plot the state of the robot
    plt.plot(stateHistory[:, 0], stateHistory[:, 1], 'k',
             label='Robot State', linewidth=4)

    plt.xlim(-1, 5)
    plt.ylim(-1, 3)
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():
    # prob1()
    prob2()

if __name__ == "__main__":
    main()
