import socket
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 7778

def send_dcs_bios_command(cmd):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto((cmd + "\n").encode(), (UDP_IP, UDP_PORT))
    sock.close()
    print(f"Sent: {cmd}")

# Wait for 15 seconds, then bank left for 2 seconds
print("Waiting 15 seconds...")
time.sleep(15)

print("Banking left...")
send_dcs_bios_command("FC3_PLANE_ROLL_LEFT 1")  # press the key
time.sleep(2)                                     # hold it for 2 seconds
send_dcs_bios_command("FC3_PLANE_ROLL_LEFT 0")  # release the key

print("Maneuver complete. Check DCS!")