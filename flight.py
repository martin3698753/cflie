import logging
import time
import read

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    joystick = read.main()
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        # Arm the Crazyflie
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            time.sleep(0.1)
        print('Taking off')

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)
        print('In air')

        # for _ in range(30):
        #     x, y = read.read(joystick)
        #     print(x, y)
        #     cf.commander.send_hover_setpoint(x, y, 0, 0.4)
        #     time.sleep(0.1)
        height = 0.4
        while not (read.stop(joystick)):
            c = 8
            height += read.up_down(joystick)
            x, y, z = read.read(joystick)
            cf.commander.send_zdistance_setpoint(x*c, y*c, z*c, height)
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
            cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
            time.sleep(0.1)

        cf.commander.send_stop_setpoint()
        # Hand control over to the high level commander to avoid timeout and locking of the Crazyflie
        cf.commander.send_notify_setpoint_stop()
