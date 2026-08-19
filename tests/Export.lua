-- Soket kütüphanelerini DCS'in kendi içinden çağırıyoruz
package.path = package.path .. ";C:/Users/aliko/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
package.cpath = package.cpath .. ";C:/Users/aliko/AppData/Roaming/luarocks/lib/lua/5.1/socket/?.dll"
local socket = require("socket")

local host = "127.0.0.1"
local port_send = 8000 -- DCS'in Python'a veri göndereceği port
local port_recv = 8001 -- DCS'in Python'dan veri alacağı port

local udp_send
local udp_recv

-- Görev başladığında bir kez çalışır
function LuaExportStart()
    udp_send = socket.udp()
    
    udp_recv = socket.udp()
    udp_recv:setsockname("*", port_recv)
    udp_recv:settimeout(0) -- Non-blocking (DCS'i dondurmaması için 0 olmalı)

    -- Python'a görevin başladığını bildir
    udp_send:sendto("EVENT: MISSION_START\n", host, port_send)
end

-- Her frame'den hemen önce çalışır (Gelen komutları okumak için idealdir)
function LuaExportBeforeNextFrame()
    local data = udp_recv:receive()
    if data then
        -- Python'dan gelen aksiyonu uygula
        if data == "ACTION:FLARE" then
            LoSetCommand(357) -- Flare (İzli mermi/Isı fişeği) at
        elseif data == "ACTION:CHAFF" then
            LoSetCommand(358) -- Chaff (Radar yanıltıcı) at
        elseif data == "ACTION:GEAR" then
            LoSetCommand(68)  -- İniş takımlarını aç/kapat
        end
    end
end

-- Her frame'den sonra çalışır (OBS toplamak için idealdir)
function LuaExportAfterNextFrame()
    local t = LoGetModelTime() -- Simülasyon içi zaman
    local selfData = LoGetSelfData() -- Kendi uçağımızın verileri

    if selfData then
        -- Önemli OBS verilerini (İrtifa ve Yön) alalım
        local alt = selfData.Position.y -- İrtifa (Metre)
        local hdg = selfData.Heading    -- Yön (Radyan)
        
        -- Veriyi formatlayıp Python'a gönderiyoruz (Saniyede onlarca kez çalışır)
        local msg = string.format("OBS: time=%.2f, alt=%.2f, hdg=%.2f\n", t, alt, hdg)
        udp_send:sendto(msg, host, port_send)
    end
end

-- Görev sonlandığında bir kez çalışır
function LuaExportStop()
    if udp_send then
        udp_send:sendto("EVENT: MISSION_TERMINATE\n", host, port_send)
        udp_send:close()
    end
    if udp_recv then
        udp_recv:close()
    end
end