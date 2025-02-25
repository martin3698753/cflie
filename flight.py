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
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.utils import power_switch

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
psw = power_switch.PowerSwitch(URI)

DEFAULT_HEIGHT = 0.4
INTERVAL = 100 #ms

x_test = np.zero(30)
t_test = np.zero(30)

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

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

def move_prep(scf):
    #IDK what this actually do
    # scf.cf.param.set_value('kalman.resetEstimation', '1')
    # time.sleep(0.1)
    # scf.cf.param.set_value('kalman.resetEstimation', '0')
    # time.sleep(2)

    # Arm the Crazyflie
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
    print('In ai, ready')

def flight(scf):
    height = DEFAULT_HEIGHT
    while not (read.stop(joystick)):
        c = 1.5
        height += read.up_down(joystick)
        x, y, z = read.read(joystick)
        scf.cf.commander.send_hover_setpoint(y*c, x*c, z*180, height) #Why I didn't use hover instead?
        if read.crit(joystick):
            psw.platform_power_down()
    print('Landing')

    # for _ in range(50):
    #     cf.commander.send_hover_setpoint(0.5, 0, -36 * 2, 0.4)
    #     time.sleep(0.1)
    #
    # for _ in range(20):
    #     cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
    #     time.sleep(0.1)
    #
    for y in range(10):
        scf.cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
        time.sleep(0.1)

    scf.cf.commander.send_stop_setpoint()
    # Hand control over to the high level commander to avoid timeout and locking of the Crazyflie
    scf.cf.commander.send_notify_setpoint_stop()


if __name__ == '__main__':
    joystick = read.main()

    # Initialize the low-level drivers
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
