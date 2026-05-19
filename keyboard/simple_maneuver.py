import socket
import time
from pynput.keyboard import Key, Controller

# ---- Configuration ----
UDP_IP = "127.0.0.1"
UDP_PORT = 5005               
TURN_HOLD_TIME = 5.0           # how long to hold the key (seconds)
STRAIGHT_TIME = 15.0           
# -----------------------

def main():
    # Set up UDP listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)

    keyboard = Controller()
    mission_started = False
    start_timestamp = None

    print("Waiting for DCS telemetry...")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            # data is something like "IAS: 250.0 RALT: 100.0 ..."
            if not mission_started:
                mission_started = True
                start_timestamp = time.time()
                print(f"Mission detected. Flying straight for {STRAIGHT_TIME} seconds...")

            # Check if we have passed the straight flight time
            if mission_started and (time.time() - start_timestamp) >= STRAIGHT_TIME:
                print("Time's up! Executing left turn...")
                keyboard.press(Key.left)
                time.sleep(TURN_HOLD_TIME)    # hold the key
                keyboard.release(Key.left)
                keyboard.press(Key.down)
                time.sleep(TURN_HOLD_TIME)    # hold the key
                keyboard.release(Key.down)
                keyboard.press(Key.right)
                time.sleep(TURN_HOLD_TIME)    # hold the key
                keyboard.release(Key.right)

                print("Turn completed. Exiting.")
                break

        except socket.timeout:
            continue

    sock.close()

if __name__ == "__main__":
    main()