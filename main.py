import logging
import sys
import os
import time
import numpy as np
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.4

deck_attached_event = Event()

logging.basicConfig(level=logging.DEBUG)

def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        mc.forward(1, velocity=1)
        mc.stop()
        mc.up(0.3, velocity=2)
        time.sleep(1)
        mc.back(2, velocity=1)
        mc.stop()
        mc.down(0.4, velocity=2)
        mc.land()


def acc_callback(timestamp, data, logconf):
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
    print(data)


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
        batconf.add_variable('pm.batteryLevel', 'float')
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
