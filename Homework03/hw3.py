import math
from shutil import SameFileError
from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import jax.numpy as jnp
from pathlib import Path

# Global variables
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
def main():
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

if __name__ == "__main__":
    main()
