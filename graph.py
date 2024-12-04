import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd
import pickdir
import maketab as mt
import makedata as md


if __name__ == '__main__':
    # path_dir = pickdir.choose_directory('data')
    path_dir = 'data/26-11-24/'
    battery = mt.battery(path_dir)
    t = mt.time(path_dir)
    md.data(battery)
    # position = readcsv(path_dir+'position.csv')
    # acceleration = readcsv(path_dir+'acceleration.csv')
    # t = acceleration[0]
    # power = acceleration[1]*position[1]*MASS
    # fig, axs = plt.subplots(2, layout='constrained')
    # axs[0].plot(t, power, label='power')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Power (Watt)')
    # axs[0].set_title('Power consumption')
    # axs[1].plot(t, battery[1], label='batV')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Voltage (V)')
    # axs[1].set_title('Battery voltage consumption')
    # plt.show()
    # pos3d(path_dir+'position.csv')
