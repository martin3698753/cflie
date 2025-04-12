import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import os
import glob
import pandas as pd
import pickdir
import maketab as mt
from sklearn import preprocessing
#import lstm

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
    #path_dir = pickdir.choose_directory('data')+'/'
    #path_dir = 'data/26-11-24/'
    path_dir = 'data/9-4-25/'
    battery = mt.battery(path_dir)
    #t = np.arange(0,battery.shape[1]*100, 100)*0.1
    t = mt.time(path_dir)
    me = mt.power(path_dir)
    en = mt.sum_ar(me)
    print(en[-1])
    en = en/2600
    #pred = mt.prediction(path_dir)
    #print(pred.shape, battery.shape)


    # motor = mt.readcsv(path_dir+'motor.csv')
    # motor = motor = (motor/65535)*100
    # thr = mt.thrust(path_dir)
    # av = mt.ang_vel(path_dir)
    # #me = thr*0.05*av*0.1
    # me = ((thr[1]/4)*av[1] + (thr[2]/4)*av[2] + (thr[3]/4)*av[3] + (thr[4]/4)*av[4])*0.047*0.1*0.05
    # mech = np.sum(me)

    x, y, z = mt.position_graph(path_dir)

    # work = norm(work)
    # battery = norm(battery)
    #work = mt.sum_ar(energy)
    # work = norm(work)
    # x, y = window(power, battery)
    # neuron.train_lin(x, y)
    # pred = neuron.predict_lin(power)
    # plt.figure(figsize=(8, 6))
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.plot(t, z)
    # plt.plot(t, av[1], label='m1 RPM')
    # plt.plot(t, av[2], label='m2 RPM')
    # plt.plot(t, av[3], label='m3 RPM')
    # plt.plot(t, av[4], label='m4 RPM')
    plt.plot(t, me, label='Výkon (W)')
    plt.plot(t, en)
    # plt.plot(t, energy, label='battery energy (J)')
    # plt.plot(t, work, label='work (J)')
    #plt.plot(t[:len(pred)], pred)
    plt.plot(t, battery[1], label='baterie (V)')
    # plt.plot(t, mt.sum_ar(me), label='Energie (J)')
    #plt.plot(t, mech_pred, label='61*t-92')
    #plt.scatter(t, pred, s=.5, label='prediction')
    #plt.ylim(2.5,4.5)
    #plt.xlim(50000, 60000)
    plt.xlabel("čas t(s)", fontsize=12)
    #plt.text(1, 1, ('Čas letu byl '+str(round(t[-1]/60000, 3))+' min\n'+'Energie = '+str(round(mech, 3))+' J'))
    # For example hover doesn't have position data
    dist = 0
    try:
        dist = mt.position(path_dir)
    except:
        pass
    #plt.legend(title=('Čas letu byl '+str(round(t[-1]/60000, 3))+' min\n'+'Energie = '+str(round(mech, 3))+' J\n'+ 'Uletěná vzdálenost = ' +str(round(dist,3)))+' m')
    #plt.legend(title=('A = '+str(round(slope))+'\n'+'B = '+str(round(intercept))))
    plt.show()
