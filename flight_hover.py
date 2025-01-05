import logging
import sys
import os
import time
import numpy as np
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.4
deck_attached_event = Event()

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def move(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        while True:
            try:
                pass
            except KeyboardInterrupt:
                mc.land()

def acc_callback(timestamp, data, logconf):
    print(data)

    filename = logconf.name+'.csv'
    names = np.array(list(data.items()))
    names = names[:,0]

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

def create_file(filename):
    if os.path.exists(filename):
        print(f"File '{filename}' already exists")
        print(f"File '{filename}' already exists")
        print(f"File '{filename}' already exists")
    else:
        with open(filename, 'w') as f:
            pass
        print(f"File '{filename}' created")

if __name__ == '__main__':
    #Create files
    create_file('acceleration.csv')
    create_file('position.csv')
    create_file('battery.csv')

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=param_deck_flow)
        time.sleep(1)


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

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        acconf.start()
        posconf.start()
        batconf.start()
        move(scf)
        acconf.stop()
        posconf.stop()
        batconf.stop()
