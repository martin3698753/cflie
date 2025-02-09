import logging
from cfclient.utils.config import Config
from cfclient.utils.input import JoystickReader
import time
import numpy as np

# Set up logging to see what's happening
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def main():
    # Create an instance of JoystickReader
    joystick_reader = JoystickReader(do_device_discovery=True)

    # Find available devices
    available_devices = joystick_reader.available_devices()

    if not available_devices:
        print("No joystick devices found!")
        return

    # Print out available devices
    for device in available_devices:
        print(f"Found device: {device.name}")
        time.sleep(1)

    # Select the first available device
    selected_device = available_devices[0]
    device_name = selected_device.name

    joystick_reader.enableRawReading(device_name)
    joystick_reader.start_input(device_name)

    return joystick_reader

    # try:
    #     # Enable raw reading to get input values
    #     #joystick_reader.enableRawReading(device_name)
    #
    #     # Optional: Check if there's a saved mapping for this device
    #     # saved_mapping = joystick_reader.get_saved_device_mapping(device_name)
    #     # if saved_mapping:
    #     #     print(f"Recommended mapping for {device_name}: {saved_mapping}")
    #
    #     # Start reading input from the device
    #     #joystick_reader.start_input(device_name)
    #
    #     # Read raw values
    #     while True:
    #         try:
    #             # Read raw values from the input device
    #             axes, buttons, mapped_values = joystick_reader.read_raw_values()
    #
    #             # Print out the current state of axes and buttons
    #             print("Axes: %s", axes)
    #             print("Buttons: %s", buttons)
    #
    #             # Optional: Add a way to break the loop, e.g., time-based or key interrupt
    #         except KeyboardInterrupt:
    #             break
    #         time.sleep(0.5)
    # except Exception as e:
    #     print(f"Error reading joystick: {e}")
    #
    # finally:
    #     # Stop raw reading and close the device
    #     joystick_reader.stop_raw_reading()

def read(joystick_reader):
    axes, buttons, mapped_values = joystick_reader.read_raw_values()
    # print(axes[0]) #left right of left stick
    # print(axes[1]) #up down of left stick
    # print(axes[3]) #left right of right stick
    # print(axes[4]) #up down of right stick
    # print('----------------------------------------------')
    return round(-axes[0], 1), round(-axes[1], 1), round(axes[3], 1)

def stop(joystick_reader):
    axes, buttons, mapped_values = joystick_reader.read_raw_values()
    if(buttons[5]==1):
        return True
    else:
        return False

def crit(joystick_reader):
    axes, buttons, mapped_values = joystick_reader.read_raw_values()
    if (buttons[1] == 1):
        return True
    else:
        return False

def up_down(joystick_reader):
    axes, buttons, mapped_values = joystick_reader.read_raw_values()
    if (axes[5] > 0):
        return 0.001
    elif (axes[2] > 0):
        return -0.001
    else:
        return 0

# if __name__ == "__main__":
#     joystick = main()
#     while(True):
#         read(joystick)
#         time.sleep(0.5)
