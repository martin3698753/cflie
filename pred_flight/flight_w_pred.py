import logging
import sys
import os
import time
import numpy as np
from threading import Event
import random
import datetime
import read

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.utils import power_switch

from batpred import BatSeqModel

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
psw = power_switch.PowerSwitch(URI)

DEFAULT_HEIGHT = 0.6
INTERVAL = 100 #ms

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

model = BatSeqModel()
latest_voltage = None
latest_pwm = None
latest_timestamp = None

def acc_callback(timestamp, data, logconf):
    global latest_voltage, latest_pwm, latest_timestamp
    filename = logconf.name+'.csv'
    names = np.array(list(data.items()))
    names = names[:,0]

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('time,' + ','.join(names) + '\n')

    with open(filename, 'a') as f:
        f.write(str(timestamp) + ',' + ','.join(str(data[n]) for n in names) + '\n')

    if logconf.name == 'battery':
        latest_voltage = data['pm.vbat']
        latest_timestamp = timestamp
    elif logconf.name == 'motor':
        pwm_values = [data['motor.m1'], data['motor.m2'], data['motor.m3'], data['motor.m4']]
        latest_pwm = sum(pwm_values) / 4 / 65535  # normalize to [0, 1]

    if latest_voltage is not None and latest_pwm is not None and latest_timestamp is not None:
        model.pred(latest_voltage, latest_pwm, latest_timestamp)
        latest_voltage, latest_pwm, latest_timestamp = None, None, None

def move_prep(scf):
    scf.cf.platform.send_arming_request(True)
    print("Armed")
    time.sleep(1.0)

    for y in range(10):
        scf.cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
        time.sleep(0.1)
    print('Taking off')

    for _ in range(20):
        scf.cf.commander.send_hover_setpoint(0, 0, 0, DEFAULT_HEIGHT)
        time.sleep(0.1)
    print('In air, ready')

def flight(scf):
    height = DEFAULT_HEIGHT
    while not (read.stop(joystick)):
        c = 2
        height += read.up_down(joystick)
        x, y, z = read.read(joystick)
        scf.cf.commander.send_hover_setpoint(y*c, x*c, z*180, height)
        if read.crit(joystick):
            psw.platform_power_down()
    print('Landing')

    for y in range(10):
        scf.cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
        time.sleep(0.1)

    scf.cf.commander.send_stop_setpoint()
    scf.cf.commander.send_notify_setpoint_stop()

if __name__ == '__main__':
    joystick = read.main()

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        posconf = LogConfig(name='position', period_in_ms=INTERVAL)
        posconf.add_variable('stateEstimate.x', 'float')
        posconf.add_variable('stateEstimate.y', 'float')
        posconf.add_variable('stateEstimate.z', 'float')
        scf.cf.log.add_config(posconf)
        posconf.data_received_cb.add_callback(acc_callback)

        logconf = LogConfig(name='motor', period_in_ms=INTERVAL)
        logconf.add_variable('motor.m1', 'uint16_t')
        logconf.add_variable('motor.m2', 'uint16_t')
        logconf.add_variable('motor.m3', 'uint16_t')
        logconf.add_variable('motor.m4', 'uint16_t')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(acc_callback)

        batconf = LogConfig(name='battery', period_in_ms=INTERVAL)
        batconf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(batconf)
        batconf.data_received_cb.add_callback(acc_callback)

        posconf.start()
        logconf.start()
        batconf.start()

        move_prep(scf)
        flight(scf)

        posconf.stop()
        logconf.stop()
        batconf.stop()
