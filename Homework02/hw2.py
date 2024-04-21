from venv import create
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from torch import randint
from scipy.stats import entropy

# Setting the print length of numpy
np.set_printoptions(edgeitems=30, linewidth=100000)

# Creating the arena
arena = np.zeros((25, 25))

# Creating a default source location
defaultSource = np.array([7, 19])

# Source
source = defaultSource

# Current robot position
robot = randint(0, 25, [2])

# Creating a list of locations visted
visitedLocations = np.zeros((25, 25))
visitedLocations[robot[0]][robot[1]] = 1

# Chances grid based on the source location
chancesGrid = np.zeros((25, 25))
beliefGrid = np.ones((25, 25))*(1.0/(25*25))

# List of positive and negative readings
positiveReadings = np.empty((0, 2))
negativeReadings = np.empty((0, 2))

#### Initializaiton functions ####
# Create the chances grid based on the source location
def createChancesGrid(sourceIn):
    source = sourceIn
    
    # Source location
    chancesGrid[source[0]][source[1]] = 1

    # 1/2 locations
    chancesGrid[source[0]-1:source[0]+2, source[1]-1] = 0.5
    chancesGrid[source[0]-1:source[0]+2, source[1]+1] = 0.5

    # 1/3 locations
    chancesGrid[source[0]-2:source[0]+3, source[1]-2] = 1.0/3.0
    chancesGrid[source[0]-2:source[0]+3, source[1]+2] = 1.0/3.0

    # 1/4 locations
    chancesGrid[source[0]-3:source[0]+4, source[1]-3] = 0.25
    chancesGrid[source[0]-3:source[0]+4, source[1]+3] = 0.25

# Create a source location at a random point
def createRandomSourceLocation():
    global source
    source = np.random.randint(3, 22, size=2)

    createChancesGrid(source)

###### Bernoulli functions ######
# Return the bernoulli chances of getting a positive value based on
# a given point and the source
def bernoulliChances(point, sourceIn):
    # print("source: ", sourceIn)
    # print("point: ", point)
    if (abs(point[0] - sourceIn[0]) > 3 or abs(point[1] - sourceIn[1]) > 3):
        return 0.0
    else:
        if (abs(point[0] - sourceIn[0]) == 0 and abs(point[1] - sourceIn[1]) == 0):
            return 1.0
        elif (abs(point[0] - sourceIn[0]) <= 1 and abs(point[1] - sourceIn[1]) == 1):
            return 0.5
        elif (abs(point[0] - sourceIn[0]) <= 2 and abs(point[1] - sourceIn[1]) == 2):
            return 1.0/3.0
        elif (abs(point[0] - sourceIn[0]) <= 3 and abs(point[1] - sourceIn[1]) == 3):
            return 0.25
        else:
            return 0.0

# Measure the current point the robot is at
def bernoulli():
    # Get the chances of the current point reading a positive reading
    global source
    chances = bernoulliChances(robot, source)

    # Get a random number
    randNum = np.random.rand()

    # Check if the random number is less than the chances
    if (randNum < chances):
        # Positive reading
        global positiveReadings
        positiveReadings = np.append(positiveReadings, [robot], axis=0)
        return 1
    else:
        # Negative reading
        global negativeReadings
        negativeReadings = np.append(negativeReadings, [robot], axis=0)
        return 0
   
# Generate an array of bernoulli chances
def bernoulliChancesArray(point):
    array = np.zeros((25, 25))
    for i in range(0, 25):
        for j in range(0, 25):
            array[i][j] = bernoulliChances(point, np.array([i, j]))

    return array

# Generate a negative bernoulli array
def negativeBernoulliArray(point):
    array = np.zeros((25, 25))
    for i in range(0, 25):
        for j in range(0, 25):
            array[i][j] = 1 - bernoulliChances(point, np.array([i, j]))

    return array

###### Actions the robots can take ######
# Move the robot to the left
def moveLeft():
    global robot
    if (robot[0] > 0):
        robot = np.array([robot[0]-1, robot[1]])
        visitedLocations[robot[0]][robot[1]] = 1
        updateBeliefGrid(bernoulli())

# Move the robot to the right
def moveRight():
    global robot
    if (robot[0] < 24):
        robot = np.array([robot[0]+1, robot[1]])
        visitedLocations[robot[0]][robot[1]] = 1
        updateBeliefGrid(bernoulli())
    
# Move the robot up
def moveUp():
    global robot
    if (robot[1] < 24):
        robot = np.array([robot[0], robot[1]+1])
        visitedLocations[robot[0]][robot[1]] = 1
        updateBeliefGrid(bernoulli())

# Move the robot down
def moveDown():
    global robot
    if (robot[1] > 0):
        robot = np.array([robot[0], robot[1]-1])
        visitedLocations[robot[0]][robot[1]] = 1
        updateBeliefGrid(bernoulli())
        
# Stay still
def stayStill():
    visitedLocations[robot[0]][robot[1]] = 1
    updateBeliefGrid(bernoulli())

# Get the integral term of the grid
def getIntegralTerm(reading):
    # Getting the integral term
    if (reading == 1):
        return jnp.sum(jnp.multiply(beliefGrid, reading))
    else:
        return jnp.sum(jnp.multiply(beliefGrid, 1 - reading))

# Update the probability grid based on a reading
def updateBeliefGrid(reading):
    # Getting the current integral term
    integralTerm = getIntegralTerm(reading)

    # Update the belief grid
    global beliefGrid, robot
    
    # Updating the belief grid based on the reading
    if (reading == 1):
        beliefGrid = jnp.divide(jnp.multiply(beliefGrid, bernoulliChancesArray(robot)), integralTerm)
    else:
        beliefGrid = jnp.divide(jnp.multiply(beliefGrid, 1 - bernoulliChancesArray(robot)), integralTerm)

# Create a theorectical belief grid based on a reading
def createTheoreticalBeliefGrid(reading, point):
    # Getting the integral term
    integralTerm = getIntegralTerm(reading)

    # Creating a theoretical belief grid
    global beliefGrid
    theoreticalGrid = beliefGrid
    if (reading == 1):
        theoreticalGrid = jnp.divide(jnp.multiply(beliefGrid, bernoulliChancesArray(point)), integralTerm)
    else:
        theoreticalGrid = jnp.divide(jnp.multiply(beliefGrid, 1 - bernoulliChancesArray(point)), integralTerm)

    return theoreticalGrid

# Perform a number of timesteps
def performTimesteps(numTimesteps):
    for i in range(0, numTimesteps):
        break

####### Plotting functions #######
# Plot the visited locations
def plotVisitedGrid(axIndexes):
    # Getting the current ax
    curAx = ax[axIndexes[0]][axIndexes[1]]
    
    # Return the plot
    curAx.imshow(np.transpose(visitedLocations), cmap='Oranges', interpolation='nearest',
                 extent=[-0.5, 24.5, -0.5, 24.5], zorder=1, origin='lower',
                 alpha=0.5, label='Visited Locations')
    
    curAx.scatter([], [], color='orange', zorder=1,
                  s=75, label='Locations Visited', marker='s', alpha=0.5)

# Plot the belief grid
def plotBeliefGrid(axIndexes):
    # Getting the current ax
    curAx = ax[axIndexes[0]][axIndexes[1]]
    
    # Return the plot
    return curAx.imshow(np.transpose(beliefGrid), cmap='Greens', interpolation='nearest',
                        extent=[-0.5, 24.5, -0.5, 24.5], zorder=1, origin='lower', alpha=0.65)

# Display the robot on the grid
def displayRobot(axIndexes):
    # Getting the current ax
    curAx = ax[axIndexes[0]][axIndexes[1]]
    # Robot point
    global robotPlot
    robotPlot = curAx.scatter(robot[0], robot[1], color='g', zorder=1,
                              s=75, label='Robot')

# Plot the chances grid
def plotChancesGrid(axIndexes):
    # Getting the current ax
    curAx = ax[axIndexes[0]][axIndexes[1]]

    # Set the aspect and the limits
    curAx.set_aspect('equal')
    curAx.set_xlim(0, 25)
    curAx.set_ylim(0, 25)
    
    # Return the plot
    return curAx.imshow(np.transpose(chancesGrid), cmap='binary', interpolation='nearest',
                                                 extent=[-0.5, 24.5, -0.5, 24.5], zorder=0, origin='lower')

# My algorithm
def myExploreAlgorithm():
    moveUp()
    moveUp()
    moveUp()
    moveRight()
    moveRight()
    moveDown()
    moveDown()
    moveDown()
    moveRight()
    moveRight()

# Infotaxis algorithm
def infoTaxisAlgorithm():
    # Getting global variables
    global beliefGrid, robot

    # Setting the J values to inf
    upJ = np.inf
    downJ = np.inf
    leftJ = np.inf
    rightJ = np.inf
    stayStillJ = np.inf

    # Calculations for each direction

    # Only calculate if the robot can move up
    if (robot[1] < 24):
        # Up
        upJ = calculateJValue(np.array([robot[0], robot[1]+1]))
    print("upJ: ", upJ)

    # Only calculate if the robot can move down
    if (robot[1] > 0):
        # Down
        downJ = calculateJValue(np.array([robot[0], robot[1]-1]))
    print("downJ: ", downJ)

    # Only calculate if the robot can move left
    if (robot[0] > 0):
        # Left
        leftJ = calculateJValue(np.array([robot[0]-1, robot[1]]))
    print("leftJ: ", leftJ)

    # Only calculate if the robot can move right
    if (robot[0] < 24):
        # Right
        rightJ = calculateJValue(np.array([robot[0]+1, robot[1]]))
    print("rightJ: ", rightJ)

    # Find J for staying still
    stayStillJ = calculateJValue(robot) + 1.0
    print("stayStillJ: ", stayStillJ)

    # Get the min value of the J values
    minJ = min(upJ, downJ, leftJ, rightJ, stayStillJ)
    print("minJ: ", minJ)

    # Move the robot based on the max value
    if (minJ == upJ):
        print("Moving up")
        moveUp()
    elif (minJ == downJ):
        print("Moving down")
        moveDown()
    elif (minJ == leftJ):
        print("Moving left")
        moveLeft()
    elif (minJ == rightJ):
        print("Moving right")
        moveRight()
    else:
        print("Staying still")
        stayStill()

    print()

# Calculate the J(x) value for a given point
def calculateJValue(point):
    # Get the current entropy of the belief grid
    currentEntropy = entropy(beliefGrid.ravel())

    # Compute the entropy of the belief grid at the point given a positive reading
    positiveEntropy =  entropy(createTheoreticalBeliefGrid(1, point).ravel())

    # Compute the entropy of the belief grid at the point given a negative reading
    negativeEntropy = entropy(createTheoreticalBeliefGrid(0, point).ravel())

    # Get the weighted sum of the two entropies
    weightedPosEntropy = currentEntropy - jnp.multiply(beliefGrid, positiveEntropy)
    weightedNegEntropy = currentEntropy - jnp.multiply(beliefGrid, negativeEntropy)

    # Get the sum of the two weighted entropies
    sumWeightedEntropy = jnp.sum(weightedPosEntropy) + jnp.sum(weightedNegEntropy)

    return sumWeightedEntropy

######### Problem functions #########
def prob1():
    # Create a figure
    global fig, ax 
    fig, ax = plt.subplots(2, 5)
    fig.set_label("Problem 1")
    fig.canvas.manager.set_window_title("Homework 2")

    # Create the chances grid
    createRandomSourceLocation()

    # Update the belief grid
    visitedLocations[robot[0]][robot[1]] = 1
    updateBeliefGrid(bernoulli())

    # For loop to display robot after 10 timesteps
    curPoint = 0
    for i in range(0, 2):
        for j in range(0, 5):
            # # Do the infotaxis algorithm x times
            x = 10
            for k in range(0, x):
                # Infotaxis algorithm
                infoTaxisAlgorithm()
                curPoint += 1

            # Getting the current ax coods
            curAx = np.array([i, j])

            # Display the locations visited
            plotVisitedGrid(curAx)

            # Plot the belief grid
            beliefGridPlot = plotBeliefGrid(curAx)

            # Display robot on grid
            displayRobot(curAx)

            # Plot the chances grid
            chancesGridPlot = plotChancesGrid(curAx)

            # Set the title of the plot
            ax[i][j].set_title(f"Plot after {curPoint} points")

            # Setting the bounds of the plot
            ax[i][j].set_xlim(-0.5, 24.5)
            ax[i][j].set_ylim(-0.5, 24.5)

            # Show the grid
            ax[i][j].set_xticks(np.linspace(0.5, 23.5, 24), minor=True)
            ax[i][j].set_yticks(np.linspace(0.5, 23.5, 24), minor=True)
            ax[i][j].grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.25)

    fig.legend(handles=ax[0][0].get_legend_handles_labels()[0], loc='upper right')
    # fig.colorbar(chancesGridPlot, ax=ax, label='Chances of a positive reading')
    fig.colorbar(beliefGridPlot, ax=ax, label='Belief in a positive reading')

    plt.show()

# Main function
def main():
    prob1()

if __name__ == "__main__":
    main()
