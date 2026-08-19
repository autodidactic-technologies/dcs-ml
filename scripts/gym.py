import socket
import time
import threading
import os

UDP_IP = "127.0.0.1"
PORT_EXPORT_RECV = 8000
PORT_EXPORT_SEND = 8001
PORT_MISSION_RECV = 8002
PORT_MISSION_SEND = 8003

# Soket Kurulumları
sock_export_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_export_recv.bind((UDP_IP, PORT_EXPORT_RECV))

sock_mission_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_mission_recv.bind((UDP_IP, PORT_MISSION_RECV))

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# RL Değişkenleri
blue_agent_alive = True
red_agent_alive = True
total_reward_blue = 0
total_reward_red = 0
mission_ended = False

def listen_export_obs():
    """Sürekli akan hız, irtifa gibi state verilerini toplar"""
    while not mission_ended:
        try:
            data, _ = sock_export_recv.recvfrom(1024)
            msg = data.decode('utf-8').strip()
        except:
            pass

def check_end_condition():
    """Mavi takımın -500 puana ulaşıp ulaşmadığını kontrol eder"""
    global mission_ended
    if total_reward_blue <= -500:
        print(f"\n[!!!] GÖREV BİTTİ: Mavi Takım {total_reward_blue} puana ulaştı.")
        sock_send.sendto("ACTION: END_MISSION".encode('utf-8'), (UDP_IP, PORT_MISSION_SEND))
        mission_ended = True
        # Python scriptini sonlandırmak için (opsiyonel):
        time.sleep(2) # Mesajın DCS'e gitmesi için kısa bir bekleme
        os._exit(0)

def listen_mission_events():
    """Hasar ve Ölüm gibi kesikli olayları yakalar"""
    global blue_agent_alive, red_agent_alive, total_reward_blue, total_reward_red

    print("Mission Event dinleyicisi aktif...")
    while not mission_ended:
        try:
            data, _ = sock_mission_recv.recvfrom(1024)
            msg = data.decode('utf-8').strip()

            if "EVENT: HIT" in msg:
                total_reward_blue -= 10
                total_reward_red -= 10
                print(f"[-] Ajan Hasar Aldı! Mavi Takım Güncel Ödül: {total_reward_blue}")
                check_end_condition()

            elif "EVENT: DEAD" in msg:
                total_reward_blue -= 100
                total_reward_red -= 100
                blue_agent_alive = False
                red_agent_alive = False
                print(f"[X] Ajan Öldü! Mavi Takım Güncel Ödül: {total_reward_blue}. Respawn tetikleniyor...")
                check_end_condition()

            elif "EVENT: RESPAWN_DONE" in msg:
                print("[+] Ajanlar yeniden doğdu! Yeni Episode başlıyor.\n")
                # Eğer ödüllerin sıfırlanmasını istiyorsan alttaki 2 satırı aktif tut. 
                # Ama toplam ödülü episode'lar boyunca biriktirmek istiyorsan bunları silmelisin.
                # (Mantıken -500'e ulaşmak için kümülatif tutman gerekiyor, o yüzden bunları yoruma aldım)
                
                # total_reward_blue = 0
                # total_reward_red = 0
                
                blue_agent_alive = True
                red_agent_alive = True

        except:
            pass

def rl_training_loop():
    """RL algoritmasının çalışacağı döngü"""
    global blue_agent_alive, red_agent_alive

    while not mission_ended:
        if not blue_agent_alive or not red_agent_alive:
            sock_send.sendto("ACTION: RESPAWN".encode('utf-8'), (UDP_IP, PORT_MISSION_SEND))
            time.sleep(1)
            continue

        time.sleep(0.1)

if __name__ == "__main__":
    t1 = threading.Thread(target=listen_export_obs, daemon=True)
    t2 = threading.Thread(target=listen_mission_events, daemon=True)
    t3 = threading.Thread(target=rl_training_loop, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        while not mission_ended:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nÇıkılıyor...")