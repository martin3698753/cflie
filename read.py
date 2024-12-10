import logging
from cfclient.utils.config import Config
from cfclient.utils.input import JoystickReader
import time

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    try:
        # Enable raw reading to get input values
        joystick_reader.enableRawReading(device_name)

        # Optional: Check if there's a saved mapping for this device
        saved_mapping = joystick_reader.get_saved_device_mapping(device_name)
        if saved_mapping:
            print(f"Recommended mapping for {device_name}: {saved_mapping}")

        # Start reading input from the device
        joystick_reader.start_input(device_name)

        # Read raw values
        while True:
            try:
                # Read raw values from the input device
                axes, buttons, mapped_values = joystick_reader.read_raw_values()

                # Print out the current state of axes and buttons
                print("Axes: %s", axes)
                print("Buttons: %s", buttons)

                # Optional: Add a way to break the loop, e.g., time-based or key interrupt
            except KeyboardInterrupt:
                break
            time.sleep(0.5)
    except Exception as e:
        print(f"Error reading joystick: {e}")

    finally:
        # Stop raw reading and close the device
        joystick_reader.stop_raw_reading()

if __name__ == "__main__":
    main()
