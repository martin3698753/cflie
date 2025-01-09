import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd

g = 9.81
TIME_INTERVAL = 0.1

def sum_ar(ar):
    new_ar = np.zeros(ar.size)
    for i in range(ar.size):
        new_ar[i] = np.sum(ar[:i+1])
    return new_ar

def window(power, battery, n):
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

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)

    # Calculate the total distance
    total_distance = np.sum(distances)
    print('Distance = ',total_distance)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x, y, z, c=t, cmap='viridis')
    #
    # ax.scatter(x[0], y[0], z[0], color='red', s=100, label='Start')
    # ax.scatter(x[-1], y[-1], z[-1], color='blue', s=100, label='End')

    # Print the total distance
    # ax.text2D(0.05, 0.95, f"Total Distance: {total_distance:.2f}", transform=ax.transAxes)

    # ax.legend()
    # plt.show()


def readcsv(filename):
    df = pd.read_csv(filename)
    df = df.iloc[:, :-1] #deleting last column cause was empty and also deleting time column
    colar = [df[col].values for col in df.columns]
    colar = np.array(colar)

    n = colar[0][0] - np.mod(colar[0][0],10)
    colar[0] = colar[0] - n

    return colar

def position(dirname):
    colar = readcsv(dirname+'position.csv')
    for i in range(1,4):
        dx = np.diff(colar[i]) #delta position
        colar[i] = np.concatenate(([0], dx)) #Keep number of dim same

    return colar[1], colar[2], colar[3]

# def dist(dirname):
#     d,_ = position(dirname)
#     dist = np.sum(d)
#     dist = round(dist, 3)
#     return dist

def velocity(dirname):
    posx, posy, posz = position(dirname)
    t = TIME_INTERVAL
    velx = posx/t
    vely = posy/t
    velz = posz/t
    return velx, vely, velz

def acceleration(dirname):
    colar = readcsv(dirname+'acceleration.csv')
    accx = colar[1]*g
    #accy = colar[2]*g
    #accz = colar[3]*g
    return accx#, accy, accz

def battery(dirname):
    bat = readcsv(dirname+'battery.csv')
    return bat[1]

def time(dirname):
    time = readcsv(dirname+'battery.csv')
    return time[0]

def thrust(dirname):
    motor = readcsv(dirname+'motor.csv')
    thr = np.zeros_like(motor)
    motor = (motor/65535)*100
    thr = (0.409*10**(-3))*motor**2 + (140.5*10**(-3))*motor - 0.099
    return (thr)

def ang_vel(dirname):
    motor = readcsv(dirname+'motor.csv')
    rmp = np.zeros_like(motor)
    motor = (motor/65535)*100
    rpm = (-1.43*10**(-1))*motor**2 + 358.12*motor + 2072.87
    av = rpm*2*np.pi/60
    return (av)

def power(dirname):
    mass = 0.05
    az = acceleration(dirname)
    F = mass*az
    w = ang_vel(dirname)*2*np.pi/60
    return(F*w)

def energy(dirname):
    posx, posy, posz = position(dirname)
    velx, vely, velz = velocity(dirname)
    m = 0.05 #kg
    KEx = 0.5*m*(np.power(velx,2))
    KEy = 0.5*m*(np.power(vely,2))
    KEz = 0.5*m*(np.power(velz,2))
    PE = m*g*posz
    return KEx, KEy, KEz, PE

# def work(dirname):
#     acc = acceleration(dirname)
#     pos = position(dirname)
#     t = 0.01 #10 ms
#     mass = 0.05
#     #return (mass*acc*pos)/t
#     return mass*acc*pos
