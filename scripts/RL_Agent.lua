package.path = "C:/Users/abc/AppData/Roaming/luarocks/share/lua/5.1/?.lua;C:/Users/abc/AppData/Roaming/luarocks/share/lua/5.1/?/init.lua;" .. package.path
package.cpath = "C:/Users/abc/AppData/Roaming/luarocks/lib/lua/5.1/?.dll;" .. package.cpath
local socket = require("socket")

local host = "127.0.0.1"
local port_send_mission = 8002 -- DCS'in Python'a Event göndereceği port
local port_recv_mission = 8003 -- Python'un DCS'e Respawn komutu göndereceği port

local udp_send = socket.udp()
local udp_recv = socket.udp()
udp_recv:setsockname("*", port_recv_mission)
udp_recv:settimeout(0)

-- 1. HASAR VE ÖLÜM (EVENT) YAKALAYICI
local RlEventHandler = {}
function RlEventHandler:onEvent(event)
    -- Mavi ve Kırmızı takımların uçaklarını kontrol et
    local blueAgent = "RL_Agent_Blue"
    local redAgent = "RL_Agent_Red"

    if event.id == world.event.S_EVENT_HIT then
        if (event.target and (event.target:getName() == blueAgent or event.target:getName() == redAgent)) then
            udp_send:sendto("EVENT: HIT\n", host, port_send_mission)
        end
    elseif event.id == world.event.S_EVENT_DEAD or event.id == world.event.S_EVENT_CRASH then
        if (event.initiator and (event.initiator:getName() == blueAgent or event.initiator:getName() == redAgent)) or
           (event.target and (event.target:getName() == blueAgent or event.target:getName() == redAgent)) then
            udp_send:sendto("EVENT: DEAD\n", host, port_send_mission)
        end
    end
end
world.addEventHandler(RlEventHandler)

-- 2. UÇAĞI DİNAMİK OLARAK YENİDEN DOĞURMA (RESPAWN) FONKSİYONU
local function getGroupData(groupName)
    for coa_name, coa_data in pairs(env.mission.coalition) do
        if coa_data.country then
            for cntry_id, cntry_data in pairs(coa_data.country) do
                for obj_type_name, obj_type_data in pairs(cntry_data) do
                    if type(obj_type_data) == "table" and obj_type_data.group then
                        for grp_id, grp_data in pairs(obj_type_data.group) do
                            if grp_data.name == groupName then return grp_data, cntry_data.id end
                        end
                    end
                end
            end
        end
    end
    return nil
end

local function RespawnAgent(agentName)
    local grpData, countryId = getGroupData(agentName)
    if grpData then
        local oldGrp = Group.getByName(agentName)
        if oldGrp then oldGrp:destroy() end
        coalition.addGroup(countryId, Group.Category.AIRPLANE, grpData)
    end
end

-- 3. PYTHON'DAN GELEN RESPAWN KOMUTUNU DİNLEME DÖNGÜSÜ
local function SocketStep()
    local data = udp_recv:receive()
    if data then
        if string.find(data, "ACTION: RESPAWN") then
            -- Her iki ajanı da respawn et
            RespawnAgent("RL_Agent_Blue")
            RespawnAgent("RL_Agent_Red")
            udp_send:sendto("EVENT: RESPAWN_DONE\n", host, port_send_mission)

	elseif string.find(data, "ACTION: END_MISSION") then
            -- DCS Ekranına mesaj gönder
            trigger.action.outText("RL Egitimi: Mavi takim -500 puana ulasti. Gorev bitiriliyor...", 15)
            
            -- ME'de (Mission Editor) yakalanmak üzere "99" numaralı bayrağı (flag) aktifleştir
            trigger.action.setUserFlag("99", 1)
        end
    end
    return timer.getTime() + 0.1
end
timer.scheduleFunction(SocketStep, {}, timer.getTime() + 1)