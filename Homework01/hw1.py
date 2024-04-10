import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

############### Creating arena ###############
# Default source location
source = np.array([0.3, 0.4])

# Colors for positive and negative sensor readings
colors = ['g', 'r']

############### Helper functions ###############
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
    newPoint = sensorLoc
    positive = False
    
    if (bernoulli(newPoint) == 1):
        positive = True
        # Checking if the arrays are empty
        if (posPoints.size == 0):
            posPoints = np.array([newPoint[0], newPoint[1]])
        else:
            posPoints = np.vstack([posPoints, newPoint])

    else:
        positive = False
        # Checking if the arrays are empty
        if (negPoints.size == 0):
            negPoints = np.array([newPoint[0], newPoint[1]])
        else:
            negPoints = np.vstack([negPoints, newPoint])

    return posPoints, negPoints, newPoint, positive

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

    # Plotting the likelyhood function
    likelyhoodPlot = ax.imshow(np.transpose(likelyhoodGrid), cmap='binary_r', interpolation='nearest',
                               extent=[0, 1, 0, 1], zorder=0, origin='lower')
        
    return likelyhoodGrid, likelyhoodPlot

# Get integral term
def getIntegralTerm(probabilityGrid, newPoint, positive):
    integralTerm = 0
    for i in range(0, probabilityGrid.shape[0]):
        for j in range(0, probabilityGrid.shape[1]):
            if (positive):
                integralTerm += bernoulliChances(newPoint, np.array([i / probabilityGrid.shape[0],
                                                                     j / probabilityGrid.shape[1]])) * probabilityGrid[i][j]
            else:
                integralTerm += (1 - bernoulliChances(newPoint, np.array([i / probabilityGrid.shape[0],
                                                                          j / probabilityGrid.shape[1]]))) * probabilityGrid[i][j]

    return integralTerm

# Create the probability function
def updateProbabilityGrid(ax, probabilityGrid, newPoint, positive):
    # Getting enumerationFactor
    enumerationFactor = probabilityGrid.shape[0]

    # Getting the integral term
    integralTerm = getIntegralTerm(probabilityGrid, newPoint, positive)
    
    # Calculating the probability function
    curCell = np.array([0.0, 0.0])
    for i in range(0, enumerationFactor):
        # Storing the x value
        curCell[0] = i / enumerationFactor

        for j in range(0, enumerationFactor):
            # Storing the y value
            curCell[1] = j / enumerationFactor
            
            if (positive):
                # Update the probability grid
                probabilityGrid[i][j]  = (bernoulliChances(newPoint, curCell) * probabilityGrid[i][j]) / integralTerm
            else:
                # Update the probability grid
                probabilityGrid[i][j]  = ((1 - bernoulliChances(newPoint, curCell)) * probabilityGrid[i][j]) / integralTerm

    probabilityPlot = ax.imshow(np.transpose(probabilityGrid), cmap='binary_r', interpolation='nearest',
                                extent=[0, 1, 0, 1], zorder=0, origin='lower')
    probabilityPlot.norm.autoscale([0, np.max(probabilityGrid)])

    return probabilityGrid, probabilityPlot

# Create a blank probability funciton
def createBlankProbabilityFunction(enumerationFactor):
    probabilityGrid = np.ones((enumerationFactor, enumerationFactor))  / (enumerationFactor**2)

    return probabilityGrid

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
    ax.scatter(sourceGuess[0], sourceGuess[1], color='orange', zorder=2,
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

    # Source guess plot
    if (plotSourceGuess):
        ax.scatter([], [], color='orange', zorder=2, marker='X', label='Source Guess', s=75)
    
    # Positive label
    ax.scatter([], [], color=colors[0], zorder=1, label='Positive', s=10)
    
    # Negative label
    ax.scatter([], [], color=colors[1], zorder=1, label='Negative', s=10)

# Create grid sample locations
def createGridSampleLocations(x, y):
    sensLocations = np.empty((0, 2))
    for i in range(0, x + 1):
        for j in range(0, y + 1):
            if (i != 0 and j != 0):
                sensLocations = np.vstack([sensLocations,
                                           np.array([i / (x + 1),
                                                     j / (y + 1)])])
                
    return sensLocations

# Main funciton
def prob1():
    # Telling the user which problem is running
    print("Problem 1 and 2")

    # Creating the random points and sorting them based on the sensor readings
    numPoints = 100
    posPoints, negPoints, points = createPoints(numPoints)
    
    # Create the plot
    fig, ax = plt.subplots()
    fig.suptitle("Problem 2")
    fig.canvas.manager.set_window_title("Homework 1")

    # Creating background for the plot
    createBackground(ax)

    # Creating circle patches to show chances
    createChancesVisual(ax, 400)

    # Adding the points to the plot
    addPointsToPlot(ax, posPoints, negPoints)

    # Creating the likelyhood function based on the points
    likelyhoodGrid, likelyhoodPlot = createLikelyhoodFunction(ax, posPoints, negPoints, 100)

    # Add the source to the plot
    addSource(ax)
    
    # Create legend
    createLegend(ax)

    # Plot the graph
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.legend(loc='upper right').set_zorder(10)
    plt.colorbar(likelyhoodPlot, ax=ax, label='Likelyhood')
    plt.show()

def prob3():
    # Telling the user what problem is running
    print("Problem 3")
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

        # Adding a source to the plot
        addSource(ax[i])

        # Adding the points to the plot
        addPointsToPlot(ax[i], posPoints, negPoints)

        # Creating the likelyhood function based on the points
        likelyhoodGrid, likelihoodPlot = createLikelyhoodFunction(ax[i], posPoints, negPoints, 50)

        # Plot the graph
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].set_aspect('equal')

        # Tell user done with this plot
        print("Plot ", i, " done")
    
    # Adding legend
    createLegend(ax[0])
    fig.legend(loc='upper right').set_zorder(10)
    fig.colorbar(likelihoodPlot, ax=ax, label='Likelyhood')

    plt.show()

def prob4():
    # Telling the user which problem is running
    print("Problem 4")

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
    
    # Probability grid
    probabilityGrid = createBlankProbabilityFunction(100)

    # For loop to create the plots
    curPointCount = 0
    for i in range(0, 2):
        for j in range(0, int(numPoints / 2)):
            # Creating background for the plot
            createBackground(ax[i][j])

            # Adding a source to the plot
            addSource(ax[i][j])
            
            # Adding the points to the plot
            addPointsToPlot(ax[i][j], posPoints, negPoints)

            # Adding a single point to the plot
            posPoints, negPoints, newPoint, positive = createSinglePoint(posPoints, negPoints, sensor)

            # Add new point to the plot
            ax[i][j].scatter(newPoint[0], newPoint[1], color=colors[positive], zorder=1, s=10)

            # Find the most likely point
            sourceGuess = findMostLikely(probabilityGrid)            

            # Create the point of the most likely source point
            addSourceGuess(ax[i][j], sourceGuess)

            # Creating the likelyhood function based on the points
            probabilityGrid, probabilityPlot = updateProbabilityGrid(ax[i][j], probabilityGrid, newPoint, positive)

            # Plot the graph
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_ylim(0, 1)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_title("Point " + str(curPointCount + 1))

            # Tell user done with this plot
            curPointCount += 1
            print("Plot ", curPointCount, " done")

    createLegend(ax[0][0], True)
    fig.legend(loc='upper right').set_zorder(10)
    fig.colorbar(probabilityPlot, ax=ax, label='Probability')
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
    
    # Probability grid
    probabilityGrid = createBlankProbabilityFunction(100)
    
    # Creating the sensor locations                
    sensLocations = createGridSampleLocations(3, 3)

    # For loop to create the plots
    curPointCount = 0
    for i in range(0, 3):
        for j in range(0, int(numPoints / 3)):
            # Creating background for the plot
            createBackground(ax[i][j])

            # Adding a source to the plot
            addSource(ax[i][j])

            # Adding the points to the plot
            addPointsToPlot(ax[i][j], posPoints, negPoints)

            # Adding a single point to the plot
            posPoints, negPoints, newPoint, positive = createSinglePoint(posPoints, negPoints, sensLocations[curPointCount])

            # Add new point to the plot
            ax[i][j].scatter(newPoint[0], newPoint[1], color=colors[positive], zorder=1, s=10)

            # Creating the likelyhood function based on the points
            probabilityGrid, probabilityPlot = updateProbabilityGrid(ax[i][j], probabilityGrid, newPoint, positive)

            # Plot the most likely point
            sourceGuess = findMostLikely(probabilityGrid)
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
    fig.colorbar(probabilityPlot, ax=ax, label='Probability')
    plt.show()

def main():
    # prob1()
    # prob3()
    # prob4()
    prob5()

if __name__ == "__main__":
    main()
