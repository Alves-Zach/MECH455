from distro import like
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from sympy import false

############### Creating arena ###############
# Default source location
source = np.array([0.3, 0.4])

# Colors for positive and negative sensor readings
colors = ['g', 'r']

# Flags to see if scatters have been added to the legend
# With default label names
posLabelFlag  = False
posLabel = "Positive"
negLabelFlag  = False
negLabel = "Negative"
sourceLabelFlag = False
sourceLabel = "Source"

############### Helper functions
# Find the chance of a positive reading
def bernoulliChances(point, sourceIn=source):
    # Calculate the chances of reading a positive value
    chances = math.exp(-100 * (np.linalg.norm(point - sourceIn) - 0.2)**2)

    return chances

# Give a reading from the sensor
def bernoulli(point, sourceIn=source):
    # Calculate the chances of reading a positive value
    chances = bernoulliChances(point, sourceIn)

    # Roll those chances
    randNum = random.random()

    if(randNum <= chances):
        return 1
    else:
        return 0

# Create the points from random locations each time
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

    return posPoints, negPoints, points

# Create uniform points
def createUniformPoints(enumerationFactor):
    # Create 100 random points
    points = np.array([])
    
    for i in range(0, enumerationFactor):
        for j in range(0, enumerationFactor):
            if (points.size == 0):
                points = np.array([i / enumerationFactor, j / enumerationFactor])
            else:
                points = np.vstack([points, np.array([i / enumerationFactor, j / enumerationFactor])])
        print(points)
    
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

    return posPoints, negPoints, points

# Create a single point
def createSinglePoint(posPoints, negPoints, sensorLoc):
    # Create a random point
    point = sensorLoc
    
    if (bernoulli(point) == 1):
        # Checking if the arrays are empty
        if (posPoints.size == 0):
            posPoints = np.array([point[0], point[1]])
        else:
            posPoints = np.vstack([posPoints, point])

    else:
        # Checking if the arrays are empty
        if (negPoints.size == 0):
            negPoints = np.array([point[0], point[1]])
        else:
            negPoints = np.vstack([negPoints, point])

    return posPoints, negPoints

# Create points from a set point
def createPointsFromSetSpot(numPoints, point):
    # Creating the list of positive points and negative points
    posPoints = np.empty((0, 2))
    negPoints = np.empty((0, 2))

    # Create lists of points that result in positive or negative reading
    for i in range(0, numPoints):
        if (bernoulli(point) == 1):
            # Checking if the arrays are empty
            if (posPoints.size == 0):
                posPoints = point
            else:
                posPoints = np.vstack([posPoints, point])
                
        else:
            # Checking if the arrays are empty
            if (negPoints.size == 0):
                negPoints = point
            else:
                negPoints = np.vstack([negPoints, point])

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
def addPointsToPlot(ax, posPoints, negPoints):
    # Default params for the points
    size = 10

    # Error checking on adding scatters to the plot
    if(posPoints.size > 2):
        ax.scatter(posPoints[:, 0], posPoints[:, 1],
                   color=colors[0], zorder=1,
                   label='_nolegend_', s=size)
    if(posPoints.size == 2):
        ax.scatter(posPoints[0], posPoints[1],
                   color=colors[0], zorder=1,
                   label='_nolegend_', s=size)

    if(negPoints.size > 2):
        ax.scatter(negPoints[:, 0], negPoints[:, 1],
                   color=colors[1], zorder=1,
                   label='_nolegend_', s=size)
    if(negPoints.size == 2):
        ax.scatter(negPoints[0], negPoints[1],
                   color=colors[1], zorder=1,
                   label='_nolegend_', s=size)

# Create the likelyhood function
def createLikelyhoodFunction(ax, posPoints, negPoints, enumerationFactor=100):
    # Creating the grid
    likelyhoodGrid = np.ones((enumerationFactor, enumerationFactor))

    # Calculating the likelyhood function
    curSource = np.array([0.0, 0.0])
    maxLikelyhood = 0
    patchesArray = np.empty((0, 1))
    for i in range(0, enumerationFactor):
        # Storing the x value
        curSource[0] = i / enumerationFactor

        for j in range(0, enumerationFactor):
            # Storing the y value
            curSource[1] = j / enumerationFactor
            
            # Two loops to go through the posPoints and the negPoints
            for k in posPoints:
                likelyhoodGrid[i][j] *= bernoulliChances(k, curSource)

            for k in negPoints:
                likelyhoodGrid[i][j] *= 1 - bernoulliChances(k, curSource)

            # Check if the likelyhood is the max
            if (likelyhoodGrid[i][j] > maxLikelyhood):
                maxLikelyhood = likelyhoodGrid[i][j]

            # Create the patch
            patchesArray = np.append(patchesArray, 
                                     patches.Rectangle((i / enumerationFactor, j / enumerationFactor),
                                     1 / enumerationFactor, 1 / enumerationFactor,
                                     alpha=likelyhoodGrid[i][j], color='green',
                                     fill=True, label='_nolegend_',
                                     zorder=0))

    # Normalize the likelyhood function and add it to the plot
    for i in patchesArray:
        i.set_alpha(i.get_alpha() / maxLikelyhood)
        ax.add_patch(i)
        
    return likelyhoodGrid

# Create the background for the plot
def createBackground(ax):
    # Create the patch to be the background
    rect = patches.Rectangle((0, 0), 1, 1, facecolor='black',
                             alpha=0.9, zorder=-1, label='_nolegend_')
    ax.add_patch(rect)

# Add a source to the plot
def addSource(ax):
    # Add the source to the plot
    ax.scatter(source[0], source[1], color='royalblue', zorder=2,
               label='_nolegend_', s=75, marker='X')

# Add source guess to the plot
def addSourceGuess(ax, sourceGuess):
    # Add the source to the plot
    ax.scatter(sourceGuess[0], sourceGuess[1], color='yellow', zorder=2,
               label='_nolegend_', s=75, marker='X')

# Find the location of the highest likelyhood
def findMostLikely(likelyhoodGrid):
    maxLikelyhood = 0
    sourceGuess = np.array([0, 0])
    for k in range(0, likelyhoodGrid.shape[0]):
        for l in range(0, likelyhoodGrid.shape[1]):
            if (likelyhoodGrid[k][l] > maxLikelyhood):
                maxLikelyhood = likelyhoodGrid[k][l]
                sourceGuess = np.array([k / likelyhoodGrid.shape[0],
                                        l / likelyhoodGrid.shape[1]])
    
    return sourceGuess

# Create legend
def createLegend(ax, plotSourceGuess=False):
    # Source label
    ax.scatter([], [], color='royalblue', zorder=2, marker='X', label='Source', s=75)
    
    # Positive label
    ax.scatter([], [], color=colors[0], zorder=1, label='Positive', s=10)
    
    # Negative label
    ax.scatter([], [], color=colors[1], zorder=1, label='Negative', s=10)

    # Source guess plot
    if (plotSourceGuess):
        ax.scatter([], [], color='yellow', zorder=2, marker='X', label='Source Guess', s=75)

# Main funciton
def prob1():
    # Creating the random points and sorting them based on the sensor readings
    numPoints = 100
    posPoints, negPoints, points = createPoints(numPoints)
    
    # Create the plot
    fig, ax = plt.subplots()
    fig.suptitle("Problem 1 and 2")
    fig.canvas.manager.set_window_title("Homework 1")

    # Creating background for the plot
    createBackground(ax)

    # Creating circle patches to show chances
    createChancesVisual(ax, 400)

    # Adding the points to the plot
    addPointsToPlot(ax, posPoints, negPoints)

    # Creating the likelyhood function based on the points
    createLikelyhoodFunction(ax, posPoints, negPoints, 100)

    # Add the source to the plot
    addSource(ax)

    # Plot the graph
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.legend(loc='upper right').set_zorder(10)
    plt.show()

def prob3():
    # Creating 100 points at the same location
    numPoints = 100

    # Create the plot
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Problem 3")
    fig.canvas.manager.set_window_title("Homework 1")

    # For loop to create the plots
    for i in range(0, 3):
        # Choosing the location of the sensor
        sensor = np.random.rand(2)

        # Loop to test that location numPoints times
        posPoints, negPoints = createPointsFromSetSpot(numPoints, sensor)

        # Creating background for the plot
        createBackground(ax[i])

        # Creating circle patches to show chances
        createChancesVisual(ax[i], 400)

        # Adding a source to the plot
        addSource(ax[i])

        # Adding the points to the plot
        addPointsToPlot(ax[i], posPoints, negPoints)

        # Creating the likelyhood function based on the points
        createLikelyhoodFunction(ax[i], posPoints, negPoints, 50)

        # Plot the graph
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].set_aspect('equal')
        plt.legend(loc='upper right').set_zorder(10)

        # Tell user done with this plot
        print("Plot ", i, " done")

    plt.show()

def prob4():
    # Creating 10 points at the same location
    numPoints = 10

    # Create the plot
    fig, ax = plt.subplots(2, 5)
    fig.suptitle("Problem 4")
    fig.canvas.manager.set_window_title("Homework 1")

    # Choosing the location of the sensor
    # sensor = np.random.rand(2)
    sensor = np.array([0.5, 0.55])

    # Creating empty positive and negative points
    posPoints = np.empty((0, 2))
    negPoints = np.empty((0, 2))

    # For loop to create the plots
    curPointCount = 0
    for i in range(0, 2):
        for j in range(0, int(numPoints / 2)):
            # Creating background for the plot
            createBackground(ax[i][j])

            # Creating circle patches to show chances
            createChancesVisual(ax[i][j], 300)

            # Adding a source to the plot
            addSource(ax[i][j])

            # Adding a single point to the plot
            posPoints, negPoints = createSinglePoint(posPoints, negPoints, sensor)

            # Adding the points to the plot
            addPointsToPlot(ax[i][j], posPoints, negPoints)

            # Creating the likelyhood function based on the points
            likelyhoodGrid = createLikelyhoodFunction(ax[i][j], posPoints, negPoints, 50)

            # Find the most likely point
            sourceGuess = findMostLikely(likelyhoodGrid)            

            # Create the point of the most likely source point
            addSourceGuess(ax[i][j], sourceGuess)

            # Plot the graph
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_ylim(0, 1)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_title("Point " + str(curPointCount + 1))

            # Tell user done with this plot
            curPointCount += 1
            print("Plot ", curPointCount, " done")

    # handles, labels = ax[0, 0].get_legend_handles_labels()
    createLegend(ax[0][0], True)
    fig.legend(loc='upper right').set_zorder(10)
    plt.show()

def prob5():
    # Creating 9 points at the same location
    numPoints = 9

    # Create the plot
    fig, ax = plt.subplots(3, 3)
    fig.suptitle("Problem 5")
    fig.canvas.manager.set_window_title("Homework 1")

    # Creating empty positive and negative points
    posPoints = np.empty((0, 2))
    negPoints = np.empty((0, 2))

    # Points to test
    sensorLocations = np.array([[0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
                               [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
                               [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]])
    
    # Creating the sensor locations
    sensorEnumeration = 3
    sensLocations = np.empty((0, 2))
    for i in range(0, sensorEnumeration + 1):
        for j in range(0, sensorEnumeration + 1):
            if (i != 0 and j != 0):
                sensLocations = np.vstack([sensLocations,
                                           np.array([i / (sensorEnumeration + 1),
                                                     j / (sensorEnumeration + 1)])])

    # For loop to create the plots
    curPointCount = 0
    for i in range(0, 3):
        for j in range(0, int(numPoints / 3)):
            # Creating background for the plot
            createBackground(ax[i][j])

            # Creating circle patches to show chances
            createChancesVisual(ax[i][j], 300)

            # Adding a source to the plot
            addSource(ax[i][j])

            # Adding a single point to the plot
            posPoints, negPoints = createSinglePoint(posPoints, negPoints, sensorLocations[curPointCount])

            # Adding the points to the plot
            addPointsToPlot(ax[i][j], posPoints, negPoints)

            # Creating the likelyhood function based on the points
            likelyhoodGrid = createLikelyhoodFunction(ax[i][j], posPoints, negPoints, 50)

            # Plot the most likely point
            sourceGuess = findMostLikely(likelyhoodGrid)
            addSourceGuess(ax[i][j], sourceGuess)

            # Plot the graph
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_ylim(0, 1)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_title("Point " + str(curPointCount + 1))

            # Tell user done with this plot
            curPointCount += 1
            print("Plot ", curPointCount, " done")

    # handles, labels = ax[0, 0].get_legend_handles_labels()
    createLegend(ax[0][0], True)
    fig.legend(loc='upper right').set_zorder(10)
    plt.show()

def main():
    # prob4()
    prob5()

if __name__ == "__main__":
    main()
