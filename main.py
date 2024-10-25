import logging
import sys
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

DEFAULT_HEIGHT = 0.7
title = False

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        for i in range(5):
            mc.forward(distance_m=1, velocity=0.5)
            time.sleep(0.05)
            mc.turn_left(180)
            time.sleep(0.05)
            mc.forward(distance_m=1, velocity=0.5)
            time.sleep(0.05)
            mc.turn_left(180)
            time.sleep(0.05)

        mc.land()


def log_pos_callback(timestamp, data, logconf):
    print(data)
    global title
    names = np.array(list(data.items()))
    names = names[:,0]
    f = open("data.csv", "a")
    for n in names:
        if not title:
            for name in names:
                f.write(name+',')
            f.write('\n')
            title = True
        f.write(str(data[n]) + ',')
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

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')

        logconf.add_variable('pm.vbatMV', 'float')
        logconf.add_variable('pm.batteryLevel', 'float')

        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()
        move(scf)
        logconf.stop()




