import read
import time

joystick = read.main()

while(True):
    b = read.stop(joystick)
    print(b.type)
    time.sleep(0.5)
