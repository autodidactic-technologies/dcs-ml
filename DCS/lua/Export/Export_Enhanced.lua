-- Enhanced DCS Export Script for RL-LLM Integration
-- Provides comprehensive tactical data for 25-dimensional observation space

local socket = require("socket")

-- Configuration
local MySocket = nil
local IPAddress = "127.0.0.1"
local Port = 5005

-- Data collection variables
local lastUpdateTime = 0
local updateInterval = 0.1  -- 10 Hz update rate
local missionStartTime = 0

function LuaExportStart()
    log.write("Enhanced_Export", log.INFO, "🚀 Enhanced DCS Export Starting")
    
    MySocket = socket.udp()
    MySocket:setpeername(IPAddress, Port)
    
    missionStartTime = DCS.getModelTime()
    
    log.write("Enhanced_Export", log.INFO, "✅ Connected to " .. IPAddress .. ":" .. Port)
end

function LuaExportAfterNextFrame()
    if not MySocket then return end
    
    local currentTime = DCS.getModelTime()
    if currentTime - lastUpdateTime < updateInterval then
        return  -- Rate limiting
    end
    lastUpdateTime = currentTime
    
    -- Enhanced data collection
    local enhancedData = collectEnhancedTacticalData()
    
    if enhancedData then
        local jsonData = json.encode(enhancedData)
        MySocket:send(jsonData)
    end
end

function collectEnhancedTacticalData()
    -- Basic flight data
    local IAS = LoGetIndicatedAirSpeed() or 0
    local RALT = LoGetAltitudeAboveGroundLevel() or 0
    local altBar = LoGetAltitudeAboveSeaLevel() or 0
    local pitch, bank, yaw = LoGetADIPitchBankYaw()
    
    -- Enhanced tactical data
    local accelerationUnits = LoGetAccelerationUnits()
    local angularVelocity = LoGetAngularVelocity()
    local targetInfo = LoGetTargetInformation()
    local lockedTargetInfo = LoGetLockedTargetInformation()
    local twsInfo = LoGetTWSInfo()
    
    -- Construct enhanced data structure
    local tacticalData = {
        timestamp = DCS.getModelTime(),
        ego = {
            lat = 0,  -- Would need additional DCS functions
            lon = 0,  -- Would need additional DCS functions  
            altitude_sea_level = altBar,
            altitude_ground = RALT,
            speed = IAS,
            heading = yaw and math.deg(yaw) or 0,
            pitch = pitch and math.deg(pitch) or 0,
            roll = bank and math.deg(bank) or 0,
            
            -- Enhanced tactical data
            g_force = accelerationUnits and accelerationUnits.y or 1.0,
            turn_rate = angularVelocity and math.deg(angularVelocity.z) or 0,
            climb_rate = accelerationUnits and accelerationUnits.z or 0,
            
            -- Weapons and sensors
            munition_count = 4,  -- Would need weapon system integration
            target_locked = lockedTargetInfo ~= nil,
            lock_duration = 0,  -- Would need tracking
            
            -- Mission data
            mission_time = currentTime - missionStartTime,
            health = 1.0  -- Would need damage system integration
        },
        enemies = {},  -- Would need AI aircraft detection
        allies = {},   -- Would need friendly aircraft detection
        mission_context = {
            mission_type = "air_superiority",
            threat_environment = "medium"
        }
    }
    
    return tacticalData
end

function LuaExportStop()
    if MySocket then
        MySocket:close()
        log.write("Enhanced_Export", log.INFO, "✅ Enhanced Export Stopped")
    end
end

-- JSON encoding function (simplified)
json = {
    encode = function(obj)
        if type(obj) == "table" then
            local result = "{"
            local first = true
            for k, v in pairs(obj) do
                if not first then result = result .. "," end
                result = result .. '"' .. tostring(k) .. '":' .. json.encode(v)
                first = false
            end
            return result .. "}"
        elseif type(obj) == "string" then
            return '"' .. obj .. '"'
        else
            return tostring(obj)
        end
    end
}