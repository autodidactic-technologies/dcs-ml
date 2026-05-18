-- Set package paths
package.path = package.path .. ";C:/Users/aliko/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
package.cpath = package.cpath .. ";C:/Users/aliko/AppData/Roaming/luarocks/lib/lua/5.1/socket/?.dll"

-- package.path = package.path .. ";C:/Users/autodidactic/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
-- package.cpath = package.cpath .. ";C:/Users/autodidactic/AppData/Roaming/luarocks/lib/lua/5.1/?.dll" 

local socket = require("socket")

-- Define global variables
local MySocket = nil
local IPAddress = "127.0.0.1"
local Port = 5005  -- Ensure your listener is using this port

function LuaExportStart()
    log.write("Export.lua", log.INFO, "✅ Starting Lua Export")

    -- Use UDP socket (TCP not needed)
    MySocket = socket.udp()
    MySocket:setpeername(IPAddress, Port)  -- Remove the setoption line

    log.write("Export.lua", log.INFO, "✅ Connected to " .. IPAddress .. ":" .. Port)
end

function LuaExportAfterNextFrame()
    if not MySocket then return end

    local IAS = LoGetIndicatedAirSpeed() or 0
    local RALT = LoGetAltitudeAboveGroundLevel() or 0
    local AoA = LoGetAngleOfAttack() or 0
    local altBar = LoGetAltitudeAboveSeaLevel() or 0
    local pitch, bank, yaw = LoGetADIPitchBankYaw()
    pitch = pitch or 0
    bank = bank or 0
    yaw = yaw or 0

--     local tws = LoGetTWSInfo() or 0
--     local target_info = LoGetTargetInformation() or 0
--     local locked_target_info = LoGetLockedTargetInformation() or 0
--     local f15_tws = LoGetF15_TWS_Contacts() or 0
--     local sighting = LoGetSightingSystemInfo() or 0
--     local wing_targets = LoGetWingTargets() or 0
    log.write("Export.lua", log.INFO, "✅ Sending IndicatedAirSpeed: " .. IAS)
    log.write("Export.lua", log.INFO, "✅ Sending AltitudeAboveGroundLeve: " .. RALT)
    log.write("Export.lua", log.INFO, "✅ Sending AngleOfAttack: " .. AoA)
    --log.write("Export.lua", log.INFO, "✅ Sending AltitudeAboveSeaLevel: " .. altbar)
    log.write("Export.lua", log.INFO, "✅ Sending Pitch: " .. pitch)
    log.write("Export.lua", log.INFO, "✅ Sending Bank: " .. bank)
    log.write("Export.lua", log.INFO, "✅ Sending Yaw: " .. yaw)
    

--     MySocket:send(string.format("IAS: %.4f  RALT: %.4f AoA: %.4f altbar: %.4f pitch: %.4f bank: %.4f yaw: %.4f tws: %.4f target info: %.4f locked: %.4f f15: %.4f sighting: %.4f wing: %.4f\n", IAS, RALT, AoA, altBar,pitch, bank,yaw, tws, target_info, locked_target_info, f15_tws, sighting, wing_targets))
    MySocket:send(string.format("IndicatedAirSpeed: %.4f  AltitudeAboveGroundLeve: %.4f AngleOfAttack: %.4f     Pitch: %.4f Bank: %.4f Yaw: %.4f\n", IAS, RALT, AoA,pitch, bank,yaw))
end

function LuaExportStop()
    if MySocket then
        MySocket:close()
        log.write("Export.lua", log.INFO, "✅ Socket closed")
    end
end
