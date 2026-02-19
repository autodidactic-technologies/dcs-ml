import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"DCS Listener başlatıldı: {UDP_IP}:{UDP_PORT}")
print("DCS'den veri bekleniyor...\n")

try:
    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size 1024 bytes
        message = data.decode('utf-8').strip()
        print(f"Alınan veri: {message}")
        
except KeyboardInterrupt:
    print("\n\n Listener kapatılıyor...")
    sock.close()
