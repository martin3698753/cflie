import logging
import sys
import os
import time
import numpy as np
from threading import Event
import random
import datetime

import read

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.commander import Commander

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.4
position_estimate = [0,0]

deck_attached_event = Event()

current_time = time.localtime()
current_path = 'data/'+str(current_time.tm_mday)+'-'+str(current_time.tm_mon)+'-'+str(current_time.tm_hour)+'-'+str(current_time.tm_min)+'-'+str(current_time.tm_sec)

logging.basicConfig(level=logging.DEBUG)

joystick = read.main()

def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        BOX_LIMIT = 0.5
        body_x_cmd = 0.2 #velocity
        body_y_cmd = 0.1
        x = 0
        y = 0

        while (True):
            x, y = read.read(joystick)
            # max_vel = random.uniform(0.5, 1)
            # if position_estimate[0] > BOX_LIMIT:
            #     body_x_cmd=-max_vel
            # elif position_estimate[0] < -BOX_LIMIT:
            #     body_x_cmd=max_vel
            # if position_estimate[1] > BOX_LIMIT:
            #     body_y_cmd=-max_vel
            # elif position_estimate[1] < -BOX_LIMIT:
            #     body_y_cmd=max_vel
        mc.start_linear_motion(x, y, 0, 0)


def acc_callback(timestamp, data, logconf):
    global position_estimate
    try:
        position_estimate[0] = data['stateEstimate.x']
        position_estimate[1] = data['stateEstimate.y']
    except:
        pass
    #print(data)

    filename = current_path+'/'+logconf.name+'.csv'
    names = np.array(list(data.items()))
    names = names[:,0]

    if not os.path.exists(filename):
        f = open(filename, 'w')
        f.write('time,')
        for n in names:
            f.write(n+',')
        f.write('\n')
        f.close()

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
        print("Data path already exist, wait another minute LOL")
        sys.exit()

    #Drivers init
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='acceleration', period_in_ms=10)
        logconf.add_variable('acc.x', 'float')
        logconf.add_variable('acc.y', 'float')
        logconf.add_variable('acc.z', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(acc_callback)

        posconf = LogConfig(name='position', period_in_ms=10)
        posconf.add_variable('stateEstimate.x', 'float')
        posconf.add_variable('stateEstimate.y', 'float')
        posconf.add_variable('stateEstimate.z', 'float')
        scf.cf.log.add_config(posconf)
        posconf.data_received_cb.add_callback(acc_callback)

        batconf = LogConfig(name='battery', period_in_ms=10)
        batconf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(batconf)
        batconf.data_received_cb.add_callback(acc_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()
        posconf.start()
        batconf.start()
        move(scf)
        logconf.stop()
        posconf.stop()
        batconf.stop()
