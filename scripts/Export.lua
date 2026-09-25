-- ==========================================================
-- Su-25T cold start + auto takeoff, with debug logging
-- Put this file at: Saved Games\DCS\Scripts\Export.lua
-- Log file will appear at: Saved Games\DCS\Logs\Su25AutoTakeoff.log
-- ==========================================================

local phase = "wait"
local startTime = nil
local phaseTime = nil      -- when the current phase started
local aboveSince = nil     -- when RPM first went above the "running" mark
local lastLogTime = 0
local log_file = nil

-- tuning values, test and change if needed
local IDLE_THRUST      = 1.0   -- guess for idle position (thrust axis is inverted, -1=full)
local FULL_THRUST      = -1.0  -- thrust is inverted, -1 = full power
local RPM_RUN_THRESHOLD = 30   -- percent. Real idle RPM is ~33%, NOT 90%
local RPM_HOLD_TIME    = 3     -- seconds RPM must stay above threshold to count as "running"
local ROTATE_SPEED     = 70    -- m/s (~252 km/h), speed to pull nose up
local CLIMB_PITCH      = -0.25 -- pitch stick input (flip sign if wrong way)
local GEAR_UP_ALT      = 15    -- meters, when to raise gear
local SAFE_ALT         = 150   -- meters, when script hands control back
local START_TIMEOUT    = 45    -- seconds to wait for engines before retry

local function log(msg)
    if log_file then
        local t = LoGetModelTime() or 0
        log_file:write(string.format("[%7.1fs] %s\n", t, msg))
        log_file:flush()
    end
end

local function setPhase(newPhase)
    log(string.format("phase change: %s -> %s", phase, newPhase))
    phase = newPhase
    phaseTime = LoGetModelTime()
    aboveSince = nil
end

function LuaExportStart()
    log_file = io.open(lfs.writedir().."/Logs/Su25AutoTakeoff.log", "w")
    startTime = LoGetModelTime()
    phaseTime = startTime
    phase = "wait"
    log("script started")
end

function LuaExportBeforeNextFrame()
    local self = LoGetSelfData()
    if not self then return end
    if self.Name ~= "Su-25T" then return end

    local now = LoGetModelTime()
    local t = now - startTime

    -- print status every 0.5 seconds, so quick RPM changes are not missed
    if now - lastLogTime > 0.5 then
        local eng = LoGetEngineInfo()
        local ias = LoGetIndicatedAirSpeed() or 0
        local alt = LoGetAltitudeAboveGroundLevel() or 0
        if eng then
            log(string.format("phase=%s  RPM L=%.1f R=%.1f  IAS=%.1f  ALT=%.1f",
                phase, eng.RPM.left, eng.RPM.right, ias, alt))
        end
        lastLogTime = now
    end

    if phase == "wait" and t > 2 then
        LoSetCommand(315)          -- power on
        setPhase("power_on")

    elseif phase == "power_on" and t > 4 then
        LoSetCommand(309)          -- start both engines
        setPhase("starting")

    elseif phase == "starting" then
        LoSetCommand(2004, IDLE_THRUST)   -- hold throttle at idle while spooling
        local eng = LoGetEngineInfo()
        if eng and eng.RPM.left > RPM_RUN_THRESHOLD and eng.RPM.right > RPM_RUN_THRESHOLD then
            if not aboveSince then aboveSince = now end
            if now - aboveSince > RPM_HOLD_TIME then
                LoSetCommand(75)       -- wheel brakes off
                LoSetCommand(145)      -- flaps on (takeoff setting)
                setPhase("roll")
            end
        else
            aboveSince = nil
            if now - phaseTime > START_TIMEOUT then
                log("engines still not running, retrying start")
                LoSetCommand(309)      -- try starting again
                phaseTime = now        -- reset timeout clock
            end
        end

    elseif phase == "roll" then
        LoSetCommand(2004, FULL_THRUST)
        local ias = LoGetIndicatedAirSpeed() or 0
        if ias > ROTATE_SPEED then
            setPhase("rotate")
        end

    elseif phase == "rotate" then
        LoSetCommand(2004, FULL_THRUST)
        LoSetCommand(2001, CLIMB_PITCH)
        local alt = LoGetAltitudeAboveGroundLevel() or 0
        if alt > GEAR_UP_ALT then
            LoSetCommand(430)     -- gear up
            setPhase("climb")
        end

    elseif phase == "climb" then
        LoSetCommand(2004, FULL_THRUST)
        LoSetCommand(2001, CLIMB_PITCH * 0.5)
        local alt = LoGetAltitudeAboveGroundLevel() or 0
        if alt > SAFE_ALT then
            setPhase("done")
        end
    end
end

function LuaExportStop()
    log("script stopped")
    if log_file then
        log_file:close()
        log_file = nil
    end
end
