-- Set package paths
package.path = package.path .. ";C:/Users/aliko/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
package.cpath = package.cpath .. ";C:/Users/aliko/AppData/Roaming/luarocks/lib/lua/5.1/socket/?.dll"

local socket = require("socket")

-- Helper: log one name=value pair (value can be nil)
local function log_kv(name, value)
    if value == nil then
        log.write("Export", log.INFO, name .. " = nil")
    elseif type(value) == "number" then
        log.write("Export", log.INFO, string.format("%s = %.4f", name, value))
    elseif type(value) == "boolean" then
        log.write("Export", log.INFO, string.format("%s = %s", name, tostring(value)))
    else
        log.write("Export", log.INFO, string.format("%s = %s", name, tostring(value)))
    end
end

-- Helper: log string value (e.g., modes, names)
local function log_str(name, value)
    if value == nil then
        log.write("Export", log.INFO, name .. " = nil")
    else
        log.write("Export", log.INFO, name .. " = " .. tostring(value))
    end
end

-- Helper: log a table recursively with prefix
local function log_table(prefix, tbl, depth)
    depth = depth or 0
    if depth > 3 then return end -- prevent infinite recursion
    
    if type(tbl) ~= "table" then
        log_kv(prefix, tbl)
        return
    end
    
    for k, v in pairs(tbl) do
        local key = prefix .. "_" .. tostring(k)
        if type(v) == "table" then
            log_table(key, v, depth + 1)
        elseif type(v) == "number" then
            log_kv(key, v)
        elseif type(v) == "boolean" then
            log_str(key, v)
        else
            log_str(key, v)
        end
    end
end

function LuaExportStart()
    log.write("Export", log.INFO, "=== Export started ===")
    -- Plan low-frequency event every 2 seconds
    LoCreateCoroutineActivity(1, 2.0, 2.0)
    Coroutines = {}
    Coroutines[1] = coroutine.create(LuaExportActivityNextEvent)
end

-- =============================================================================
-- HIGH FREQUENCY (every frame)
-- =============================================================================
function LuaExportAfterNextFrame()
    -- ---------------------------------------------------------
    -- 1. Time & Identity
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- TIME & IDENTITY ---")
    log_kv("ModelTime", LoGetModelTime())
    log_kv("MissionStartTime", LoGetMissionStartTime())
    log_str("PilotName", LoGetPilotName())
    log_kv("PlayerPlaneId", LoGetPlayerPlaneId())

    -- ---------------------------------------------------------
    -- 2. Basic Flight
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- BASIC FLIGHT ---")
    log_kv("IndicatedAirSpeed", LoGetIndicatedAirSpeed())
    log_kv("TrueAirSpeed", LoGetTrueAirSpeed())
    log_kv("AltitudeASL", LoGetAltitudeAboveSeaLevel())
    log_kv("AltitudeAGL", LoGetAltitudeAboveGroundLevel())
    log_kv("AngleOfAttack", LoGetAngleOfAttack())
    log_kv("VerticalVelocity", LoGetVerticalVelocity())
    log_kv("MachNumber", LoGetMachNumber())
    log_kv("MagneticYaw", LoGetMagneticYaw())
    log_kv("GlideDeviation", LoGetGlideDeviation())
    log_kv("SideDeviation", LoGetSideDeviation())
    log_kv("SlipBallPosition", LoGetSlipBallPosition())
    log_kv("BasicAtmPressure", LoGetBasicAtmospherePressure())

    -- ADI (returns 3 values: pitch, bank, yaw)
    local p, b, y = LoGetADIPitchBankYaw()
    log_kv("ADI_Pitch", p)
    log_kv("ADI_Bank", b)
    log_kv("ADI_Yaw", y)

    -- Acceleration (returns table with x, y, z)
    local accel = LoGetAccelerationUnits()
    if accel then
        log_table("Accel", accel)
    else
        log_str("Accel", "nil")
    end

    -- ---------------------------------------------------------
    -- 3. Velocity Vectors
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- VELOCITY VECTORS ---")
    
    -- Self velocity vector
    local vel = LoGetVectorVelocity()
    if vel then
        log_table("VelWorld", vel)
    else
        log_str("VelWorld", "nil")
    end

    -- Angular velocity
    local angVel = LoGetAngularVelocity()
    if angVel then
        log_table("AngVel", angVel)
    else
        log_str("AngVel", "nil")
    end

    -- Wind velocity
    local wind = LoGetVectorWindVelocity()
    if wind then
        log_table("Wind", wind)
    else
        log_str("Wind", "nil")
    end

    -- ---------------------------------------------------------
    -- 4. Engine Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ENGINE ---")
    local eng = LoGetEngineInfo()
    if eng then
        log_table("Eng", eng)
    else
        log_str("EngineInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 5. HSI (Horizontal Situation Indicator)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- HSI ---")
    local hsi = LoGetControlPanel_HSI()
    if hsi then
        log_table("HSI", hsi)
    else
        log_str("HSI", "nil")
    end

    -- ---------------------------------------------------------
    -- 6. Navigation Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- NAVIGATION ---")
    local nav = LoGetNavigationInfo()
    if nav then
        log_table("Nav", nav)
    else
        log_str("NavigationInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 7. Route Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ROUTE ---")
    local route = LoGetRoute()
    if route then
        log_table("Route", route)
    else
        log_str("Route", "nil")
    end

    -- ---------------------------------------------------------
    -- 8. Mechanical Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- MECHANICAL ---")
    local mech = LoGetMechInfo()
    if mech then
        log_table("Mech", mech)
    else
        log_str("MechInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 9. MCP State (Master Caution Panel)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- MCP STATE ---")
    local mcp = LoGetMCPState()
    if mcp then
        log_table("MCP", mcp)
    else
        log_str("MCPState", "nil")
    end

    -- ---------------------------------------------------------
    -- 10. Snares (Countermeasures)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SNARES ---")
    local snares = LoGetSnares()
    if snares then
        log_table("Snares", snares)
    else
        log_str("Snares", "nil")
    end

    -- ---------------------------------------------------------
    -- 11. Payload Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- PAYLOAD ---")
    local payload = LoGetPayloadInfo()
    if payload then
        log_table("Payload", payload)
    else
        log_str("Payload", "nil")
    end

    -- ---------------------------------------------------------
    -- 12. Self Data
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SELF DATA ---")
    local selfData = LoGetSelfData()
    if selfData then
        log_table("Self", selfData)
    else
        log_str("SelfData", "nil")
    end

    -- ---------------------------------------------------------
    -- 13. Camera Position
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- CAMERA ---")
    local cam = LoGetCameraPosition()
    if cam then
        log_table("Camera", cam)
    else
        log_str("Camera", "nil")
    end

    -- ---------------------------------------------------------
    -- 14. Radio Beacons
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- RADIO BEACONS ---")
    local beacons = LoGetRadioBeaconsStatus()
    if beacons then
        log_table("Beacons", beacons)
    else
        log_str("Beacons", "nil")
    end
end

-- =============================================================================
-- LOW FREQUENCY (every 2 seconds)
-- =============================================================================
function LuaExportActivityNextEvent(t)
    log.write("Export", log.INFO, "=== LOW FREQ DATA (t=" .. string.format("%.2f", t) .. ") ===")
    
    -- ---------------------------------------------------------
    -- 15. Sighting System Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SIGHTING SYSTEM ---")
    local sight = LoGetSightingSystemInfo()
    if sight then
        log_table("Sight", sight)
    else
        log_str("SightingSystem", "nil")
    end

    -- ---------------------------------------------------------
    -- 16. TWS (Threat Warning System)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- TWS ---")
    local tws = LoGetTWSInfo()
    if tws then
        log_table("TWS", tws)
    else
        log_str("TWSInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 17. Target Information (all targets)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ALL TARGETS ---")
    local targets = LoGetTargetInformation()
    if targets then
        log_kv("AllTargets_Count", #targets)
        for i, trg in ipairs(targets) do
            log_table("AllTarget_"..i, trg)
        end
    else
        log_str("AllTargets", "nil")
    end

    -- ---------------------------------------------------------
    -- 18. Locked Target Information
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- LOCKED TARGETS ---")
    local locked = LoGetLockedTargetInformation()
    if locked then
        log_kv("LockedTargets_Count", #locked)
        for i, trg in ipairs(locked) do
            log_table("LockedTarget_"..i, trg)
        end
    else
        log_str("LockedTargets", "nil")
    end

    -- ---------------------------------------------------------
    -- 19. Wingmen Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- WINGMEN ---")
    local wing = LoGetWingInfo()
    if wing then
        log_kv("Wingmen_Count", #wing)
        for i, w in ipairs(wing) do
            log_table("Wingman_"..i, w)
        end
    else
        log_str("WingInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 20. Wing Targets
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- WING TARGETS ---")
    local wingTgts = LoGetWingTargets()
    if wingTgts then
        log_kv("WingTargets_Count", #wingTgts)
        for i, wt in ipairs(wingTgts) do
            if type(wt) == "table" then
                log_table("WingTarget_"..i, wt)
            else
                log_kv("WingTarget_"..i, wt)
            end
        end
    else
        log_str("WingTargets", "nil")
    end

    -- ---------------------------------------------------------
    -- 21. World Objects (HUGE - be careful!)
    -- ---------------------------------------------------------
    -- Uncomment if you really need ALL world objects:
    -- log.write("Export", log.INFO, "--- WORLD OBJECTS ---")
    -- local world = LoGetWorldObjects()
    -- if world then
    --     local count = 0
    --     for k, v in pairs(world) do
    --         count = count + 1
    --         log_table("WorldObj_"..k, v)
    --     end
    --     log_kv("WorldObjects_Total", count)
    -- else
    --     log_str("WorldObjects", "nil")
    -- end

    return t + 2.0
end

-- =============================================================================
-- Coroutine support (required by DCS)
-- =============================================================================
function CoroutineResume(index, tCurrent)
    local ok, tNext = coroutine.resume(Coroutines[index], tCurrent)
    return ok and coroutine.status(Coroutines[index]) ~= "dead"
end

-- =============================================================================
-- Stop
-- =============================================================================
function LuaExportStop()
    log.write("Export", log.INFO, "=== Export stopped ===")
end
