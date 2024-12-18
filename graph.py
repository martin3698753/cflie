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

def norm(s):
    scaler = preprocessing.MinMaxScaler()
    s = s.reshape((len(s), 1))
    scaler = scaler.fit(s)
    d = scaler.transform(s)
    return d

if __name__ == '__main__':
    #path_dir = pickdir.choose_directory('data')+'/'
    path_dir = 'data/26-11-24/'
    battery = mt.battery(path_dir)
    power = mt.power(path_dir)
    t = mt.time(path_dir)

    # power = norm(power)
    # battery = norm(battery)
    # x, y = window(power, battery)
    # neuron.train_lin(x, y)
    # pred = neuron.predict_lin(power)
    plt.plot(t, power, label='power (W)')
    plt.plot(t, battery, label='battery (V)')
    #plt.scatter(t, pred, s=.5, label='prediction')
    plt.ylim(2.5,4.5)
    #plt.xlim(50000, 60000)
    plt.xlabel("time (ms)")
    plt.legend()
    plt.show()
    #lstm.init(t, battery)

    # net = LNU()
    # result = net.train(power, battery)
    #
    # print("Model Performance:")
    # print(f"Mean Squared Error: {result['mse']:.4f}")
    # print(f"Mean Absolute Error: {result['mae']:.4f}")
    # net.visual(result)
