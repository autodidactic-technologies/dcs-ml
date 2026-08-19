#!/usr/bin/env python3
"""
Konteynerin İÇİNDEN çalıştırılmalı (loopback UDP).
Kullanım: python3 set_time_acceleration.py 8
"""
import json
import socket
import sys
import time

HOST = "127.0.0.1"
LISTEN_PORT = 5000   # hook script'in dinlediği port
REPLY_PORT = 5001     # hook script'in cevap gönderdiği port

def send_command(code, luaEnv="gui", timeout=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, REPLY_PORT))
    sock.settimeout(timeout)
    msg = {"id": int(time.time() * 1000) % 1000000, "type": "command", "luaEnv": luaEnv, "code": code}
    sock.sendto(json.dumps(msg).encode(), (HOST, LISTEN_PORT))
    try:
        data, _ = sock.recvfrom(65536)
        return json.loads(data.decode())
    except socket.timeout:
        return {"status": "timeout", "result": None}
    finally:
        sock.close()

def check_extension_installed(luaEnv="gui"):
    code = (
        'local newPath = ";"..lfs.writedir().."Mods\\\\Services\\\\DCS-Extensions\\\\bin\\\\?.dll" '
        'if not string.find(package.cpath, newPath, 1, true) then package.cpath = package.cpath..newPath end '
        'if not dcs_extensions then dcs_extensions = require("dcs_extensions") end '
        'return dcs_extensions ~= nil'
    )
    return send_command(code, luaEnv)

def set_acceleration(value, luaEnv="gui"):
    code = (
        'local newPath = ";"..lfs.writedir().."Mods\\\\Services\\\\DCS-Extensions\\\\bin\\\\?.dll" '
        'if not string.find(package.cpath, newPath, 1, true) then package.cpath = package.cpath..newPath end '
        'if not dcs_extensions then dcs_extensions = require("dcs_extensions") end '
        f'if dcs_extensions and dcs_extensions.setAcceleration then dcs_extensions.setAcceleration({value}) end'
    )
    return send_command(code, luaEnv)

def get_acceleration(luaEnv="gui"):
    code = (
        'local newPath = ";"..lfs.writedir().."Mods\\\\Services\\\\DCS-Extensions\\\\bin\\\\?.dll" '
        'if not string.find(package.cpath, newPath, 1, true) then package.cpath = package.cpath..newPath end '
        'if not dcs_extensions then dcs_extensions = require("dcs_extensions") end '
        'if dcs_extensions and dcs_extensions.getAcceleration then return dcs_extensions.getAcceleration() else return 0 end'
    )
    return send_command(code, luaEnv)

if __name__ == "__main__":
    envs_to_try = ["gui", "server", "mission", "export"]
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

    for env in envs_to_try:
        print(f"\n=== luaEnv = {env} ===")
        check = check_extension_installed(env)
        print("  extension yüklü mü:", check)
        if check.get("status") != "success":
            continue
        setr = set_acceleration(target, env)
        print(f"  setAcceleration({target}):", setr)
        time.sleep(0.5)
        getr = get_acceleration(env)
        print("  getAcceleration():", getr)