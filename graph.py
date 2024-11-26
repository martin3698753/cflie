import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd


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

    # Print the total distance
    ax.text2D(0.05, 0.95, f"Total Distance: {total_distance:.2f}", transform=ax.transAxes)

    ax.legend()
    plt.show()


def readcsv(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1]
    colar = [df[col].values for col in df.columns]
    colar = np.array(colar)

    n = colar[0][0] - np.mod(colar[0][0],10)
    colar[0] = colar[0] - n


    if filename=='position.csv':
        for i in range(1,3):
            dx = np.diff(colar[i]) #delta position
            dx = np.concatenate(([0], dx)) #Keep number of dim same
            colar[i] = dx

    if filename=='position.csv' or filename=='acceleration.csv':
        mag = np.power(colar[1], 2) + np.power(colar[2], 2) + np.power(colar[3], 2)
        mag = np.sqrt(mag)
        colar = np.array([colar[0], mag])

    return colar

if __name__ == '__main__':
    MASS = 0.05
    battery = readcsv('battery.csv')
    position = readcsv('position.csv')
    acceleration = readcsv('acceleration.csv')
    t = acceleration[0]
    power = acceleration[1]*position[1]*MASS*t
    fig, axs = plt.subplots(2, layout='constrained')
    fig.suptitle('Drone power consumption and battery voltage')
    axs[0].plot(t, power, label='power')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Power (Watt)')
    axs[1].plot(t, battery[1], label='batV')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage (V)')
    plt.show()
    pos3d('position.csv')



    # for filename in glob.glob('position*'):
    #     df = pd.read_csv(filename)
    #     df = df.iloc[:, :-1]
    #     colar = [df[col].values for col in df.columns]
    #
    #     n = colar[0][0] - np.mod(colar[0][0],10)
    #     colar[0] = colar[0] - n
    #
    #
    #     for i in range(1,len(colar)):
    #         # dx = np.diff(colar[i])
    #         # dx = np.concatenate(([0], dx))
    #         plt.plot(colar[0], colar[i], label=df.columns[i])
