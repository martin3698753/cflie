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
import neuron
from neuron import window
from sklearn import preprocessing
import lstm

def norm(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def denorm(normalized_data, original_min, original_max):
    """
    Denormalizes data that was previously normalized using min-max scaling.

    Args:
        normalized_data: The normalized data (NumPy array or single value).
        original_min: The minimum value of the original dataset.
        original_max: The maximum value of the original dataset.

    Returns:
        The denormalized data (NumPy array or single value).
    """

    denormalized_data = normalized_data * (original_max - original_min) + original_min
    return denormalized_data


if __name__ == '__main__':
    path_dir = pickdir.choose_directory('data')+'/'
    #path_dir = 'data/26-11-24/'
    #path_dir = 'data/28-12-0-18-3/'
    battery = mt.battery(path_dir)
    power = mt.power(path_dir)
    sp = mt.work_done(path_dir)
    # #work = mt.work_done(path_dir)
    # t = mt.time(path_dir)
    t = np.arange(0, battery.size)
    energy = 250*battery*3.6

    power = norm(power)
    battery = norm(battery)
    # work = norm(work)
    # x, y = window(power, battery)
    # neuron.train_lin(x, y)
    # pred = neuron.predict_lin(power)
    plt.text(3, 2, ('Flight time = ', battery.size*0.1/60, ' s\n', 'Consumed power = ', sp[-1], ' J'), fontsize=12, color='red')
    plt.plot(t, power, label='work (J)')
    #plt.plot(t, energy, label='battery energy (J)')
    #plt.plot(t, work, label='summed work (J)')
    plt.plot(t, battery, label='battery (V)')
    #plt.scatter(t, pred, s=.5, label='prediction')
    #plt.ylim(2.5,4.5)
    #plt.xlim(50000, 60000)
    plt.xlabel("time (s)")
    plt.legend(title=('Flight time = '+str(round(battery.size*0.1/60, 3))+' min\n'+'Consumed power = '+str(round(sp[-1], 3))+' J'))
    plt.show()
    #lstm.init(t, battery)

    # net = LNU()
    # result = net.train(power, battery)
    #
    # print("Model Performance:")
    # print(f"Mean Squared Error: {result['mse']:.4f}")
    # print(f"Mean Absolute Error: {result['mae']:.4f}")
    # net.visual(result)
