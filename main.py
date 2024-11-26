import logging
import sys
import os
import time
import numpy as np
from threading import Event
import random

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.7
position_estimate = [0,0]

deck_attached_event = Event()

logging.basicConfig(level=logging.DEBUG)

def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        BOX_LIMIT = 0.5
        body_x_cmd = 0.2 #velocity
        body_y_cmd = 0.1

        while (1):
            max_vel = random.uniform(0.5, 1)
            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd=-max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd=max_vel
            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd=-max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd=max_vel

            mc.start_linear_motion(body_x_cmd, body_y_cmd, 0)
            time.sleep(0.1)


def acc_callback(timestamp, data, logconf):
    global position_estimate
    try:
        position_estimate[0] = data['stateEstimate.x']
        position_estimate[1] = data['stateEstimate.y']
    except:
        pass
    print(data)

    filename = logconf.name+'.csv'
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
