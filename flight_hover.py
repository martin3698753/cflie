import logging
import sys
import os
import time
import numpy as np
from threading import Event
import random
import datetime

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.4

current_time = time.localtime()
current_path = 'data/'+str(current_time.tm_mday)+'-'+str(current_time.tm_mon)+'-'+str(current_time.tm_hour)+'-'+str(current_time.tm_min)+'-'+str(current_time.tm_sec)

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        while True:
            pass
        mc.land()

def acc_callback(timestamp, data, logconf):
    print(data)

    filename = current_path+'/'+logconf.name+'.csv'
    names = np.array(list(data.items()))
    names = names[:,0]
    #
    # if not os.path.exists(filename):
    #     f = open(filename, 'w')
    #     f.write('time,')
    #     for n in names:
    #         f.write(n+',')
    #     f.write('\n')
    #     f.close()

    f = open(filename, 'a')
    f.write(str(timestamp)+',')
    for n in names:
        f.write(str(data[n])+',')
    f.write('\n')
    f.close()

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')


if __name__ == '__main__':
    #Create dir with date
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    else:
        print("Data path already exist, WTF is happenin?")
        sys.exit()
    #Create csv files
    open(current_path+'/acceleration.csv', 'w').close()
    open(current_path+'/position.csv', 'w').close()
    open(current_path+'/battery.csv', 'w').close()

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        acconf = LogConfig(name='acceleration', period_in_ms=100)
        acconf.add_variable('acc.x', 'float')
        acconf.add_variable('acc.y', 'float')
        acconf.add_variable('acc.z', 'float')
        scf.cf.log.add_config(acconf)
        acconf.data_received_cb.add_callback(acc_callback)

        posconf = LogConfig(name='position', period_in_ms=100)
        posconf.add_variable('stateEstimate.x', 'float')
        posconf.add_variable('stateEstimate.y', 'float')
        posconf.add_variable('stateEstimate.z', 'float')
        scf.cf.log.add_config(posconf)
        posconf.data_received_cb.add_callback(acc_callback)

        batconf = LogConfig(name='battery', period_in_ms=100)
        batconf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(batconf)
        batconf.data_received_cb.add_callback(acc_callback)

        acconf.start()
        posconf.start()
        batconf.start()
        move(scf)
        acconf.stop()
        posconf.stop()
        batconf.stop()
