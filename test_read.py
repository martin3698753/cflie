#!/usr/bin/env python3

import read
import time

joystick = read.main()
while True:
    print(read.crit(joystick))
    time.sleep(0.3)
