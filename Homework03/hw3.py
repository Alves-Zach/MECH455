import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import jax.numpy as jnp
from pathlib import Path

# Global variables
# Image file
imagePath = Path(__file__).resolve().parent / 'lincoln.jpg'
image = jnp.divide(mpimg.imread(imagePath), 255)

# List of points sampled and their corresponding readings
samples = np.empty((0, 2))
readings = np.empty((0, 1))

###### Helper functions ######
# Get the chances of a positive reading at the given point
def getChance(point):
    # Get the darkness of the point
    posChances = image[math.floor(point[0] * image.shape[0]), math.floor(point[1] * image.shape[1])]

    return posChances

# Get the reading at a given point
def getReading(point):
    # Get the chances of a positive reading at a given point
    posChances = getChance(point)

    # Get a random number between 0 and 1
    randNum = np.random.rand()

    # If the random number is less than the chances of a positive reading, then the reading is positive
    if randNum < posChances:
        readings = 1
    else:
        readings = 0

# Get the samples from the image
def getSamples(numSamples):
    # Loop over the number of desired samples
    for i in range(0, numSamples):
        # Get a random point on the image
        point = np.random.rand(2)

        # Get a reading at that point
        getReading(point)

# Main function
def main():
    # Create a mutliplot, one for the image and one for the samples
    fig, ax = plt.subplots(1, 2)

    # Getting x number of samples on the image based on the darkness of the image
    numSamples = 1000
    getSamples(numSamples)

    # Display the image
    ax[0].imshow(image, cmap='gray', extent=[0, 1, 0, 1])
    plt.show()

if __name__ == "__main__":
    main()
