import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd

def window(power, battery, n=10):
    x, y = [], []
    for i in range(len(power) - n):
        # Create input sequence of power measurements
        x.append(power[i:i+n])
        # Target is the next battery voltage
        y.append(battery[i+n])
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(x.shape[0], -1)
    return x, y

def pos3d(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1]
    colar = [df[col].values for col in df.columns]

    n = colar[0][0] - np.mod(colar[0][0],10)
    colar[0] = colar[0] - n

    t = colar[0]
    x = colar[1]
    y = colar[2]
    z = colar[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=t, cmap='viridis')

    ax.scatter(x[0], y[0], z[0], color='red', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='blue', s=100, label='End')

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)

    # Calculate the total distance
    total_distance = np.sum(distances)
    print('Distance = ',total_distance)

    # Print the total distance
    ax.text2D(0.05, 0.95, f"Total Distance: {total_distance:.2f}", transform=ax.transAxes)

    ax.legend()
    #plt.show()


def readcsv(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1] #deleting last column cause was empty and also deleting time column
    colar = [df[col].values for col in df.columns]
    colar = np.array(colar)

    n = colar[0][0] - np.mod(colar[0][0],10)
    colar[0] = colar[0] - n

    name = filename.split('/')[-1]

    return colar

def position(dirname):
    colar = readcsv(dirname+'position.csv')
    for i in range(1,3):
        dx = np.diff(colar[i]) #delta position
        dx = np.concatenate(([0], dx)) #Keep number of dim same
        colar[i] = dx

    mag = np.power(colar[1], 2) + np.power(colar[2], 2) + np.power(colar[3], 2)
    mag = np.sqrt(mag)
    #colar = np.array([colar[0], mag])
    return mag

def acceleration(dirname):
    colar = readcsv(dirname+'acceleration.csv')
    mag = np.power(colar[1], 2) + np.power(colar[2], 2) + np.power(colar[3], 2)
    mag = np.sqrt(mag)
    return mag

def battery(dirname):
    bat = readcsv(dirname+'battery.csv')
    return bat[1]

def time(dirname):
    time = readcsv(dirname+'battery.csv')
    return time[0]

def power(dirname):
    acc = acceleration(dirname)
    pos = position(dirname)
    t = 0.01 #10 ms
    mass = 0.05
    return (mass*acc*pos)/t
