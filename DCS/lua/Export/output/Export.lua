local function log_value(name, value, depth)
    depth = depth or 0
    if depth > 3 then return end -- prevent infinite recursion

    if type(value) == "table" then
        for k, v in pairs(value) do
            log_value(name .. "_" .. tostring(k), v, depth + 1)
        end
    elseif value == nil then
        log.write("Export", log.INFO, name .. " = nil")
    elseif type(value) == "number" then
        log.write("Export", log.INFO, string.format("%s = %.4f", name, value))
    else -- boolean, string, or anything else
        log.write("Export", log.INFO, name .. " = " .. tostring(value))
    end
end

function LuaExportStart()
    log.write("Export", log.INFO, "=== EXPORT STARTED ===")
end

function LuaExportAfterNextFrame()
    -- Check what data is currently available
    log.write("Export", log.INFO, "--- EXPORT PERMISSIONS ---")
    log_value("LoIsObjectExportAllowed", LoIsObjectExportAllowed())
    log_value("LoIsSensorExportAllowed", LoIsSensorExportAllowed())
    log_value("LoIsOwnshipExportAllowed", LoIsOwnshipExportAllowed())

    -- ---------------------------------------------------------
    -- Time & Identity
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- TIME & IDENTITY ---")
    log_value("LoGetModelTime", LoGetModelTime())
    log_value("LoGetMissionStartTime", LoGetMissionStartTime())
    log_value("LoGetPilotName", LoGetPilotName())
    log_value("LoGetPlayerPlaneId", LoGetPlayerPlaneId())

    -- ---------------------------------------------------------
    -- Basic Flight
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- BASIC FLIGHT ---")
    log_value("LoGetIndicatedAirSpeed", LoGetIndicatedAirSpeed())
    log_value("LoGetTrueAirSpeed", LoGetTrueAirSpeed())
    log_value("LoGetAltitudeAboveSeaLevel", LoGetAltitudeAboveSeaLevel())
    log_value("LoGetAltitudeAboveGroundLevel", LoGetAltitudeAboveGroundLevel())
    log_value("LoGetAngleOfAttack", LoGetAngleOfAttack())
    log_value("LoGetVerticalVelocity", LoGetVerticalVelocity())
    log_value("LoGetMachNumber", LoGetMachNumber())
    log_value("LoGetMagneticYaw", LoGetMagneticYaw())
    log_value("LoGetGlideDeviation", LoGetGlideDeviation())
    log_value("LoGetSideDeviation", LoGetSideDeviation())
    log_value("LoGetSlipBallPosition", LoGetSlipBallPosition())
    log_value("LoGetBasicAtmospherePressure", LoGetBasicAtmospherePressure())

    -- ADI (returns 3 values: pitch, bank, yaw)
    local LoGetADI_Pitch, LoGetADI_Bank, LoGetADI_Yaw = LoGetADIPitchBankYaw()
    log_value("LoGetADI_Pitch", LoGetADI_Pitch)
    log_value("LoGetADI_Bank", LoGetADI_Bank)
    log_value("LoGetADI_Yaw", LoGetADI_Yaw)

    -- Acceleration (returns table with x, y, z)
    local LoGetAccelerationUnits = LoGetAccelerationUnits()
    log_value("LoGetAccelerationUnits", LoGetAccelerationUnits)

    -- ---------------------------------------------------------
    -- Velocity Vectors
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- VELOCITY VECTORS ---")
    
    -- Self velocity vector
    local LoGetVectorVelocity = LoGetVectorVelocity()
    log_value("LoGetVectorVelocity", LoGetVectorVelocity)

    -- Angular velocity
    local LoGetAngularVelocity = LoGetAngularVelocity()
    log_value("LoGetAngularVelocity", LoGetAngularVelocity)

    -- Wind velocity
    local LoGetVectorWindVelocity = LoGetVectorWindVelocity()
    log_value("LoGetVectorWindVelocity", LoGetVectorWindVelocity)

    -- ---------------------------------------------------------
    -- Engine Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ENGINE ---")
    local LoGetEngineInfo = LoGetEngineInfo()
    log_value("LoGetEngineInfo", LoGetEngineInfo)

    -- ---------------------------------------------------------
    -- HSI (Horizontal Situation Indicator)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- HSI ---")
    local LoGetControlPanel_HSI = LoGetControlPanel_HSI()
    log_value("LoGetControlPanel_HSI", LoGetControlPanel_HSI)

    -- ---------------------------------------------------------
    -- Navigation Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- NAVIGATION ---")
    local LoGetNavigationInfo = LoGetNavigationInfo()
    log_value("LoGetNavigationInfo", LoGetNavigationInfo)

    -- ---------------------------------------------------------
    -- Route Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ROUTE ---")
    local LoGetRoute = LoGetRoute()
    log_value("LoGetRoute", LoGetRoute)

    -- ---------------------------------------------------------
    -- Mechanical Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- MECHANICAL ---")
    local LoGetMechInfo = LoGetMechInfo()
    log_value("LoGetMechInfo", LoGetMechInfo)

    -- ---------------------------------------------------------
    -- MCP State (Master Caution Panel)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- MCP STATE ---")
    local LoGetMCPState = LoGetMCPState()
    log_value("LoGetMCPState", LoGetMCPState)

    -- ---------------------------------------------------------
    -- Snares (Countermeasures)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SNARES ---")
    local LoGetSnares = LoGetSnares()
    log_value("LoGetSnares", LoGetSnares)

    -- ---------------------------------------------------------
    -- Payload Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- PAYLOAD ---")
    local LoGetPayloadInfo = LoGetPayloadInfo()
    log_value("LoGetPayloadInfo", LoGetPayloadInfo)

    -- ---------------------------------------------------------
    -- Self Data
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SELF DATA ---")
    local LoGetSelfData = LoGetSelfData()
    log_value("LoGetSelfData", LoGetSelfData)

    -- ---------------------------------------------------------
    -- Radio Beacons
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- RADIO BEACONS ---")
    local LoGetRadioBeaconsStatus = LoGetRadioBeaconsStatus()
    log_value("LoGetRadioBeaconsStatus", LoGetRadioBeaconsStatus)

    -- Sighting System Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- SIGHTING SYSTEM ---")
    local LoGetSightingSystemInfo = LoGetSightingSystemInfo()
    log_value("LoGetSightingSystemInfo", LoGetSightingSystemInfo)

    -- ---------------------------------------------------------
    -- TWS (Threat Warning System)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- TWS ---")
    local LoGetTWSInfo = LoGetTWSInfo()
    log_value("LoGetTWSInfo", LoGetTWSInfo)

    -- ---------------------------------------------------------
    -- Target Information (all targets)
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- ALL TARGETS ---")
    local LoGetTargetInformation = LoGetTargetInformation()
    if LoGetTargetInformation then
        log_value("LoGetTargetInformation_Count", #LoGetTargetInformation)
        for i, trg in ipairs(LoGetTargetInformation) do
            log_value("LoGetTargetInformation_"..i, trg)
        end
    else
        log_value("LoGetTargetInformation", "nil")
    end

    -- ---------------------------------------------------------
    -- Locked Target Information
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- LOCKED TARGETS ---")
    local LoGetLockedTargetInformation = LoGetLockedTargetInformation()
    if LoGetLockedTargetInformation then
        log_value("LoGetLockedTargetInformation_Count", #LoGetLockedTargetInformation)
        for i, trg in ipairs(LoGetLockedTargetInformation) do
            log_value("LoGetLockedTargetInformation_"..i, trg)
        end
    else
        log_value("LoGetLockedTargetInformation", "nil")
    end

    -- ---------------------------------------------------------
    -- Wingmen Info
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- WINGMEN ---")
    local LoGetWingInfo = LoGetWingInfo()
    if LoGetWingInfo then
        log_value("LoGetWingInfo_Count", #LoGetWingInfo)
        for i, w in ipairs(LoGetWingInfo) do
            log_value("LoGetWingInfo_"..i, w)
        end
    else
        log_value("LoGetWingInfo", "nil")
    end

    -- ---------------------------------------------------------
    -- Wing Targets
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- WING TARGETS ---")
    local LoGetWingTargets = LoGetWingTargets()
    if LoGetWingTargets then
        log_value("LoGetWingTargets_Count", #LoGetWingTargets)
        for i, wt in ipairs(LoGetWingTargets) do
            log_value("LoGetWingTargets_"..i, wt)
        end
    else
        log_value("LoGetWingTargets", "nil")
    end

    --[[
    -- ---------------------------------------------------------
    -- Object by ID
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- OBJECT BY ID ---")
    -- Test with your own aircraft (always available)
    local selfId = LoGetPlayerPlaneId()
    if selfId then
        log_value("LookingUp_ID", selfId)
        log_value("ObjectByID_Self", LoGetObjectById(selfId))
    else
        log_value("PlayerPlaneId", "nil")
    end
    --]]

    --[[
    -- ---------------------------------------------------------
    -- Camera Position
    -- ---------------------------------------------------------
    log.write("Export", log.INFO, "--- CAMERA ---")
    local LoGetCameraPosition = LoGetCameraPosition()
    log_value("LoGetCameraPosition", LoGetCameraPosition)
    --]]

    --[[
    LoGetWorldObjects
    LoGetAltitude
    LoGetCameraPosition
    LoGetNameByType
    LoGeoCoordinatesToLoCoordinates
    LoLoCoordinatesToGeoCoordinates
    LoGetHelicopterFMData
    --]]


end

-- =============================================================================
-- Stop
-- =============================================================================
function LuaExportStop()
    log.write("Export", log.INFO, "=== EXPORT STOPPED ===")
end