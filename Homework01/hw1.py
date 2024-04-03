import numpy as np
import math
import matplotlib.pyplot as plt
import random

############### Creating arena ###############
# Arena bounds
xlim = np.array([0, 1])
ylim = np.array([0, 1])

# Source location
source = np.array([0.25, 0.25])

# Colors for positive and negative sensor readings
colors = ['g', 'r']

# Give a reading from the sensor
def bernoulli(point):
    # Calculate the chances of reading a positive value
    chances = math.exp(-100 * (np.linalg.norm(point - source) - 0.2)**2)

    # print(chances)

    # Roll those chances
    randNum = random.random()

    # print(randNum)

    if(randNum <= chances):
        return 1
    else:
        return 0

# Problem 1
def prob1():
    # Create 100 random points
    points = np.random.rand(10, 2)

    # Creating the list of positive points and negative points
    posPoints = np.empty((0, 2))
    negPoints = np.empty((0, 2))

    # Create lists of points that result in positive or negative reading
    for i in points:
        print("Current point", end=": ")
        print(i)
        if (bernoulli(i) == 1):
            # Checking if the arrays are empty
            if (posPoints.size == 0):
                posPoints = np.array([i[0], i[1]])
            else:
                posPoints = np.vstack([posPoints, i])
                
            print(posPoints)
            print("Added to positive\n")
        else:
            # Checking if the arrays are empty
            if (negPoints.size == 0):
                negPoints = np.array([i[0], i[1]])
            else:
                negPoints = np.vstack([negPoints, i])
                
            print(negPoints)
            print("Added to negative\n")

    print("Negative points", end=" ")
    print(negPoints.shape, end=":\n")
    print(negPoints)
    print()

    print("Positive points", end=" ")
    print(posPoints.shape, end=":\n")
    print(posPoints)

    # Create the plot
    fig, ax = plt.subplots()

    # Checking if either set of plots is empty
    if(posPoints.size != 0):
        ax.scatter(posPoints[:, 0], posPoints[:, 1], color=colors[0])

    if(negPoints.size != 0):
        ax.scatter(negPoints[:, 0], negPoints[:, 1], color=colors[1])

    plt.show()

def main():
    prob1()

if __name__ == "__main__":
    main()
