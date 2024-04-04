import enum
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

############### Creating arena ###############
# Arena bounds
xlim = np.array([0, 1])
ylim = np.array([0, 1])

# Source location
source = np.array([0.3, 0.4])

# Colors for positive and negative sensor readings
colors = ['g', 'r']

# Find the chance of a positive reading
def bernoulliChances(point):
    # Calculate the chances of reading a positive value
    chances = math.exp(-100 * (np.linalg.norm(point - source) - 0.2)**2)

    return chances

# Give a reading from the sensor
def bernoulli(point):
    # Calculate the chances of reading a positive value
    chances = bernoulliChances(point)

    # Roll those chances
    randNum = random.random()

    if(randNum <= chances):
        return 1
    else:
        return 0

# Problem 1

# Helper functions
def createPoints(numPoints):
    # Create 100 random points
    points = np.random.rand(numPoints, 2)

    # Creating the list of positive points and negative points
    posPoints = np.empty((0, 2))
    negPoints = np.empty((0, 2))

    # Create lists of points that result in positive or negative reading
    for i in points:
        if (bernoulli(i) == 1):
            # Checking if the arrays are empty
            if (posPoints.size == 0):
                posPoints = np.array([i[0], i[1]])
            else:
                posPoints = np.vstack([posPoints, i])
                
        else:
            # Checking if the arrays are empty
            if (negPoints.size == 0):
                negPoints = np.array([i[0], i[1]])
            else:
                negPoints = np.vstack([negPoints, i])
                
    print("Negative points", end=" ")
    print(negPoints.shape, end=":\n")
    print(negPoints)
    print()

    print("Positive points", end=" ")
    print(posPoints.shape, end=":\n")
    print(posPoints)

    return posPoints, negPoints

# Create the circle patches
def createChancesVisual(ax, enumeration):
    # The loop to test the circle patches
    for i in range(0, enumeration):
        # Create the patch, and change the alpha based on the chances
        alpha = bernoulliChances(np.array([source[0] + (i / enumeration),
                                           source[1]]))
        
        circle = patches.Circle((source[0], source[1]),
                                (i / enumeration),
                                alpha=alpha,
                                color='white',
                                fill=False,
                                zorder=-1,
                                label='_nolegend_')
        ax.add_patch(circle)

# Add the points to the plot
def addPoints(ax, posPoints, negPoints, enumeration):
    # Default params for the points
    size = 10

    # Error checking on adding scatters to the plot
    if(posPoints.size > 2):
        ax.scatter(posPoints[:, 0], posPoints[:, 1],
                   color=colors[0], zorder=1,
                   label="Positive", s=size)
    if(posPoints.size == 2):
        ax.scatter(posPoints[0], posPoints[1],
                   color=colors[0], zorder=1,
                   label="Positive", s=size)

    if(negPoints.size > 2):
        ax.scatter(negPoints[:, 0], negPoints[:, 1],
                   color=colors[1], zorder=1,
                   label="Negative", s=size)
    if(negPoints.size == 2):
        ax.scatter(negPoints[0], negPoints[1],
                   color=colors[1], zorder=1,
                   label="Negative", s=size)

# Create the likelyhood function
def createLikelyhoodFunction(ax, posPoints, negPoints):
    enumerationFactor = 100
    # Create the likelyhood function
    for i in range(0, enumerationFactor + 1):
        for j in range(0, enumerationFactor + 1):
            

# Main funciton
def main():
    # Creating the random points and sorting them based on the sensor readings
    numPoints = 100
    posPoints, negPoints = createPoints(numPoints)
    
    # Create the plot
    fig, ax = plt.subplots()
    fig.suptitle("")
    fig.canvas.manager.set_window_title("Homework 1")

    # Create the patch to be the background
    rect = patches.Rectangle((0, 0), 1, 1, facecolor='black',
                             alpha=0.9, zorder=-1, label='_nolegend_')
    ax.add_patch(rect)

    # Creating circle patches to show chances
    enumeration = 500
    createChancesVisual(ax, enumeration)

    # Adding the points to the plot
    addPoints(ax, posPoints, negPoints, enumeration)

    # Creating the likelyhood function based on the points
    createLikelyhoodFunction(ax, posPoints, negPoints)

    # Adding the source to the plot
    ax.scatter(source[0], source[1], color='royalblue', zorder=2,
               label="Source", s=75, marker='X')

    # Plot the graph
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.legend(loc='upper right').set_zorder(10)
    plt.show()

if __name__ == "__main__":
    main()
