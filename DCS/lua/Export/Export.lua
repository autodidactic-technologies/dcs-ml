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
    log.write("Export", log.INFO," --- TIME & IDENTITY ---")

    log_kv("ModelTime",          LoGetModelTime())                  -- ModelTime = 0.0000 
    log_kv("MissionStartTime",   LoGetMissionStartTime())           -- MissionStartTime = 65700.0000
    log_str("PilotName",         LoGetPilotName())                  -- PilotName = New callsign
    log_kv("PlayerPlaneId",      LoGetPlayerPlaneId())              -- PlayerPlaneId = 16793344.0000

    -- ---------------------------------------------------------
    -- 2. Basic Flight
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- BASIC FLIGHT ---")

    log_kv("IndicatedAirSpeed",  LoGetIndicatedAirSpeed())         -- IndicatedAirSpeed = 132.4035
    log_kv("TrueAirSpeed",       LoGetTrueAirSpeed())              -- TrueAirSpeed = 154.1667
    log_kv("AltitudeASL",        LoGetAltitudeAboveSeaLevel())     -- AltitudeASL = 2437.5740
    log_kv("AltitudeAGL",        LoGetAltitudeAboveGroundLevel())  -- AltitudeAGL = 1414.8037
    log_kv("AngleOfAttack",      LoGetAngleOfAttack())             -- AngleOfAttack = 0.0000
    log_kv("VerticalVelocity",   LoGetVerticalVelocity())          -- VerticalVelocity = 0.0000
    log_kv("MachNumber",         LoGetMachNumber())                -- MachNumber = 0.4565
    log_kv("MagneticYaw",        LoGetMagneticYaw())               -- MagneticYaw = 8.2192
    log_kv("GlideDeviation",     LoGetGlideDeviation())            -- GlideDeviation = 0.0000
    log_kv("SideDeviation",      LoGetSideDeviation())             -- SideDeviation = 0.0000
    log_kv("SlipBallPosition",   LoGetSlipBallPosition())          -- SlipBallPosition = 0.0000
    log_kv("BasicAtmPressure",   LoGetBasicAtmospherePressure())   -- BasicAtmPressure = 760.0000

    -- ADI
    local p, b, y = LoGetADIPitchBankYaw()
    log_kv("ADI_Pitch", p)                                         -- ADI_Pitch = 0.1103
    log_kv("ADI_Bank",  b)                                         -- ADI_Bank = -0.0000
    log_kv("ADI_Yaw",   y)                                         -- ADI_Yaw = 8.3476

    -- Acceleration (3-axis)
    local accel = LoGetAccelerationUnits()
    if accel then
        log_kv("Accel_X", accel.x)                                 -- Accel_X = 0.1167
        log_kv("Accel_Y", accel.y)                                 -- Accel_Y = 0.9932
        log_kv("Accel_Z", accel.z)                                 -- Accel_Z = 0.0000
    else
        log_kv("Accel_X", nil)
        log_kv("Accel_Y", nil)
        log_kv("Accel_Z", nil)
    end

    -- Velocity vectors (world)
    local vel = LoGetVectorVelocity()
    if vel then
        log_kv("VelWorld_X", vel.x)                                -- VelWorld_X = -73.0520
        log_kv("VelWorld_Y", vel.y)                                -- VelWorld_Y = 0.0000
        log_kv("VelWorld_Z", vel.z)                                -- VelWorld_Z = 135.7600
    else
        log_kv("VelWorld_X", nil)
        log_kv("VelWorld_Y", nil)
        log_kv("VelWorld_Z", nil)
    end

    local angVel = LoGetAngularVelocity()
    if angVel then
        log_kv("AngVel_X", angVel.x)                               -- AngVel_X = 0.0000
        log_kv("AngVel_Y", angVel.y)                               -- AngVel_Y = 0.0000  
        log_kv("AngVel_Z", angVel.z)                               -- AngVel_Z = 0.0000
    else
        log_kv("AngVel_X", nil)
        log_kv("AngVel_Y", nil)
        log_kv("AngVel_Z", nil)
    end

    local wind = LoGetVectorWindVelocity()
    if wind then
        log_kv("Wind_X", wind.x)                                  -- Wind_X = -1.7145  
        log_kv("Wind_Y", wind.y)                                  -- Wind_Y = 0.0075  
        log_kv("Wind_Z", wind.z)                                  -- Wind_Z = 3.7358
    else
        log_kv("Wind_X", nil)
        log_kv("Wind_Y", nil)
        log_kv("Wind_Z", nil)
    end

    -- ---------------------------------------------------------
    -- 3. Engine
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- ENGINE ---")

    local eng = LoGetEngineInfo()
    if eng then
        log_kv("Eng_RPM_Left",        eng.RPM and eng.RPM.left)                                   --  Eng_RPM_Left = 92.2541
        log_kv("Eng_RPM_Right",       eng.RPM and eng.RPM.right)                                  --  Eng_RPM_Right = 92.2541  
        log_kv("Eng_Temp_Left",       eng.Temperature and eng.Temperature.left)                   --  Eng_Temp_Left = 635.9758
        log_kv("Eng_Temp_Right",      eng.Temperature and eng.Temperature.right)                  --  Eng_Temp_Right = 635.9758
        log_kv("Eng_HydPress_Left",   eng.HydraulicPressure and eng.HydraulicPressure.left)       --  Eng_HydPress_Left = 210.0000 
        log_kv("Eng_HydPress_Right",  eng.HydraulicPressure and eng.HydraulicPressure.right)      --  Eng_HydPress_Right = 210.0000
        log_kv("Eng_FuelCons_Left",   eng.FuelConsumption and eng.FuelConsumption.left)           --  Eng_FuelCons_Left = 0.0000
        log_kv("Eng_FuelCons_Right",  eng.FuelConsumption and eng.FuelConsumption.right)          --  Eng_FuelCons_Right = 0.0000
        log_kv("Eng_FuelInternal",    eng.fuel_internal)                                          --  Eng_FuelInternal = 3775.0000
        log_kv("Eng_FuelExternal",    eng.fuel_external)                                          --  Eng_FuelExternal = 0.0000
    else
        log_str("EngineInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- 4. HSI
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- HSI ---")

    local hsi = LoGetControlPanel_HSI()
    if hsi then
        log_kv("HSI_ADF_raw",           hsi.ADF_raw)                     -- HSI_ADF_raw = 6.2832
        log_kv("HSI_RMI_raw",           hsi.RMI_raw)                     -- HSI_RMI_raw = 6.2832
        log_kv("HSI_Heading_raw",       hsi.Heading_raw)                 -- HSI_Heading_raw = 6.2832
        log_kv("HSI_HeadingPointer",    hsi.HeadingPointer)              -- HSI_HeadingPointer = 6.2832
        log_kv("HSI_Course",            hsi.Course)                      -- HSI_Course = 0.0000
        log_kv("HSI_BearingPointer",    hsi.BearingPointer)              -- HSI_BearingPointer = 0.0000
        log_kv("HSI_CourseDeviation",   hsi.CourseDeviation)             -- HSI_CourseDeviation = 0.0000
    else
        log_str("HSI", "nil")
    end

    -- ---------------------------------------------------------
    -- 5. Mechanical / MCP
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- MECHANICAL / MCP ---")

    local mech = LoGetMechInfo()
    if mech then
        -- gear
        if mech.gear then
            log_kv("Mech_GearStatus",   mech.gear.status)                                         -- Mech_GearStatus = 0.0000
            log_kv("Mech_GearValue",    mech.gear.value)                                          -- Mech_GearValue = 0.0000
            if mech.gear.main then
                log_kv("Mech_GearLeftRod",  mech.gear.main.left and mech.gear.main.left.rod)      -- Mech_GearLeftRod = 0.0000
                log_kv("Mech_GearRightRod", mech.gear.main.right and mech.gear.main.right.rod)    -- Mech_GearRightRod = 0.0000
                log_kv("Mech_GearNoseRod",  mech.gear.main.nose and mech.gear.main.nose.rod)      -- Mech_GearNoseRod = nil
            end
        end
        log_kv("Mech_FlapsStatus",      mech.flaps and mech.flaps.status)                         -- Mech_FlapsStatus = 0.0000
        log_kv("Mech_SpeedbrakesStatus",mech.speedbrakes and mech.speedbrakes.status)             -- Mech_SpeedbrakesStatus = 0.0000
        log_kv("Mech_WingStatus",       mech.wing and mech.wing.status)                           -- Mech_WingStatus = 0.0000
        log_kv("Mech_CanopyStatus",     mech.canopy and mech.canopy.status)                       -- Mech_CanopyStatus = 0.0000
        log_kv("Mech_HookStatus",       mech.hook and mech.hook.status)                           -- Mech_HookStatus = 0.0000
        -- Control surfaces (relative -1..1)
        if mech.controlsurfaces then
            log_kv("Mech_ElevatorLeft",  mech.controlsurfaces.elevator and mech.controlsurfaces.elevator.left)      -- Mech_ElevatorLeft = -0.0000
            log_kv("Mech_ElevatorRight", mech.controlsurfaces.elevator and mech.controlsurfaces.elevator.right)     -- Mech_ElevatorRight = -0.0000
            log_kv("Mech_AileronLeft",   mech.controlsurfaces.eleron and mech.controlsurfaces.eleron.left)          -- Mech_AileronLeft = 0.0000
            log_kv("Mech_AileronRight",  mech.controlsurfaces.eleron and mech.controlsurfaces.eleron.right)         -- Mech_AileronRight = -0.2108
            log_kv("Mech_RudderLeft",    mech.controlsurfaces.rudder and mech.controlsurfaces.rudder.left)          -- Mech_RudderLeft = 0.0000
            log_kv("Mech_RudderRight",   mech.controlsurfaces.rudder and mech.controlsurfaces.rudder.right)         -- Mech_RudderRight = 0.0000
        end
    else
        log_str("MechInfo", "nil")
    end

    local mcp = LoGetMCPState()
    if mcp then
        log_str("MCP_EngineFailL",      mcp.LeftEngineFailure)          -- MCP_EngineFailL = false
        log_str("MCP_EngineFailR",      mcp.RightEngineFailure)         -- MCP_EngineFailR = false
        log_str("MCP_HydraulicsFail",   mcp.HydraulicsFailure)          -- MCP_HydraulicsFail = false
        log_str("MCP_AP_On",            mcp.AutopilotOn)                -- MCP_AP_On = false
        log_str("MCP_MasterWarning",    mcp.MasterWarning)              -- MCP_MasterWarning = false
        log_str("MCP_Stall",            mcp.StallSignalization)         -- MCP_Stall = nil 
        log_str("MCP_RadarFail",        mcp.RadarFailure)               -- MCP_RadarFail = false
        log_str("MCP_ECMFail",          mcp.ECMFailure)                 -- MCP_ECMFail = false
        log_str("MCP_FuelTankDamage",   mcp.FuelTankDamage)             -- MCP_FuelTankDamage = false
    else
        log_str("MCPState", "nil")
    end

    -- ---------------------------------------------------------
    -- 6. Snares (countermeasures left)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- SNARES ---")

    local snares = LoGetSnares()
    if snares then
        log_kv("Chaff", snares.chaff)             -- Chaff = 128.0000
        log_kv("Flare", snares.flare)             -- Flare = 128.0000
    else
        log_kv("Chaff", nil)
        log_kv("Flare", nil)
    end

    -- ---------------------------------------------------------
    -- 7. Payload (simplified: weapon counts per station)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- PAYLOAD ---")

    local payload = LoGetPayloadInfo()
    if payload then
        log_kv("Payload_CurrentStation", payload.CurrentStation)     -- Payload_CurrentStation = 0.0000
        if payload.Stations then
            for i, st in pairs(payload.Stations) do     -- Station1 = UNKNOWN_0,0,0,0 x0   Station2 = UNKNOWN_0,0,0,0 x0  Station3 = S-8KOM HEAT/Frag x20  Station4 = S-8KOM HEAT/Frag x20
                if st and st.weapon then                -- Station5 = Kh-25ML (AS-10 Karen) x1  Station6 = Kh-25ML (AS-10 Karen) x1  Station7 = 9M127 Vikhr x8  Station8 = 9M127 Vikhr x8  Station9 = FAB-250 x1  Station10 = FAB-250 x1  Station11 = UNKNOWN_0,0,0,0 x0
                    local wname = LoGetNameByType(st.weapon.level1, st.weapon.level2, st.weapon.level3, st.weapon.level4)
                    log_str("Station"..i, (wname or "unknown") .. " x" .. (st.count or 0))
                end
            end
        end
        if payload.Cannon then
            log_kv("CannonShells", payload.Cannon.shells)    -- CannonShells = 250.0000
        end
    else
        log_str("Payload", "nil")
    end

    -- ---------------------------------------------------------
    -- 8. Self Data (your own aircraft object)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO," --- SELF DATA ---")

    local selfData = LoGetSelfData()
    if selfData then
        log_str("Self_Name",            selfData.Name)         -- Self_Name = Su-25T
        log_kv("Self_Country",          selfData.Country)      -- Self_Country = 0.0000
        log_kv("Self_Coalition",        selfData.Coalition)    -- Self_Coalition = Allies
        if selfData.LatLongAlt then
            log_kv("Self_Lat",  selfData.LatLongAlt.Lat)       -- Self_Lat = 43.9717
            log_kv("Self_Lon",  selfData.LatLongAlt.Long)      -- Self_Lon = 42.7466
            log_kv("Self_Alt",  selfData.LatLongAlt.Alt)       -- Self_Alt = 2459.7347
        end
        log_kv("Self_Heading",  selfData.Heading)              -- Self_Heading = 2.0321
        log_kv("Self_Pitch",    selfData.Pitch)                -- Self_Pitch = 0.2883
        log_kv("Self_Bank",     selfData.Bank)                 -- Self_Bank = -0.1923
        if selfData.Flags then
            log_str("Self_RadarActive", selfData.Flags.RadarActive)   -- Self_RadarActive = false
            log_str("Self_Human",       selfData.Flags.Human)         -- Self_Human = true
        end
    else
        log_str("SelfData", "nil")
    end
end

-- =============================================================================
-- LOW FREQUENCY (every 2 seconds) – sensors, targets, wingmen
-- =============================================================================
function LuaExportActivityNextEvent(t)
    -- Sensors
    log.write("Export", log.INFO," --- SENSORS ---")

    local sight = LoGetSightingSystemInfo()
    if sight then
        log_str("Sight_Manufacturer", sight.Manufacturer)              -- Sight_Manufacturer = RUS
        log_str("Sight_LaunchAuth",    sight.LaunchAuthorized)         -- Sight_LaunchAuth = false
        log_str("Sight_RadarOn",       sight.radar_on)                 -- Sight_RadarOn = false
        log_str("Sight_ECMOn",         sight.ECM_on)                   -- Sight_ECMOn = false
        if sight.ScanZone then
            log_kv("Sight_ScanAzimuth",  sight.ScanZone.position and sight.ScanZone.position.azimuth)         -- Sight_ScanAzimuth = 0.0000
            log_kv("Sight_ScanElevation", sight.ScanZone.position and sight.ScanZone.position.elevation)      -- Sight_ScanElevation = 0.0000
        end
    else
        log_str("SightingSystem", "nil")
    end

    local tws = LoGetTWSInfo()
    if tws then
        log_kv("TWS_Mode", tws.Mode)                    -- TWS_Mode = 0.0000
        if tws.Emitters then
            log_kv("TWS_NumEmitters", #tws.Emitters)    -- TWS_NumEmitters = 0.0000
        end
    else
        log_str("TWSInfo", "nil")
    end

    -- Locked targets
    log.write("Export", log.INFO," --- LOCKED TARGETS ---")

    local locked = LoGetLockedTargetInformation()
    if locked then
        log_kv("LockedTargets_Count", #locked)             -- LockedTargets_Count = 0.0000
        for i, trg in ipairs(locked) do
            log_kv("Locked_"..i.."_ID", trg.ID)
            log_kv("Locked_"..i.."_Dist", trg.distance)
            log_kv("Locked_"..i.."_Closure", trg.convergence_velocity)
        end
    else
        log_str("LockedTargets", "nil")
    end

    -- Wingmen
    log.write("Export", log.INFO," --- WINGMEN ---")

    local wing = LoGetWingInfo()
    if wing then
        for i, w in ipairs(wing) do
            log_kv("Wing_"..i.."_ID", w.wingmen_id)
            log_str("Wing_"..i.."_Task", w.current_task)
        end
    else
        log_str("WingInfo", "nil")
    end

    local wingTgts = LoGetWingTargets()
    if wingTgts then
        log_kv("WingTargets_Count", #wingTgts)          -- WingTargets_Count = 0.0000
    else
        log_str("WingTargets", "nil")
    end

    -- World Objects (UNCOMMENT IF NEEDED – huge amount of data)
    -- local world = LoGetWorldObjects()
    -- if world then log_kv("WorldObjects_Count", #world) end

    return t + 2.0   -- repeat every 2 seconds
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
