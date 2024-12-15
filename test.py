import read
import time

joystick = read.main()

while(True):
    x = read.read(joystick)
    print(x)
    time.sleep(0.5)
