-- ---- names to test: change these to match your Mission Editor unit names ----
local BLUE_UNIT_NAME = "AI-1-Ace"
local RED_UNIT_NAME  = "AI-1-Veteran"
local LOG_PERIOD     = 2.0 -- seconds between test log dumps

local function log_value(name, value, depth)
    depth = depth or 0
    if depth > 6 then return end 

    if type(value) == "table" then
        for k, v in pairs(value) do
            log_value(name .. "_" .. tostring(k), v, depth + 1)
        end
    elseif value == nil then
        log.write("AIObs", log.INFO, name .. " = nil")
    elseif type(value) == "number" then
        log.write("AIObs", log.INFO, string.format("%s = %.4f", name, value))
    else -- boolean, string, enum
        log.write("AIObs", log.INFO, name .. " = " .. tostring(value))
    end
end

-- ---- logs everything on the "AI Scripting - Obs Functions" sheet for one unit ----
local function LogUnitObs(prefix, unitName)
    local unit = Unit.getByName(unitName)

    if not unit or not unit:isExist() then
        log_value(prefix, "nil (unit not found or destroyed)")
        return nil
    end

    -- Object
    log_value(prefix .. "_unit:getName", unit:getName())
    log_value(prefix .. "_unit:getCategory", unit:getCategory())
    log_value(prefix .. "_getTypeName", unit:getTypeName())
    log_value(prefix .. "_unit:getPoint", unit:getPoint())
    log_value(prefix .. "_unit:getVelocity", unit:getVelocity())
    log_value(prefix .. "_unit:inAir", unit:inAir())

    -- CoalitionObject
    log_value(prefix .. "_unit:getCoalition", unit:getCoalition())
    log_value(prefix .. "_unit:getCountry", unit:getCountry())

    -- Unit
    log_value(prefix .. "_unit:isActive", unit:isActive())
    log_value(prefix .. "_unit:getPlayerName", unit:getPlayerName())
    log_value(prefix .. "_unit:getCallsign", unit:getCallsign())
    log_value(prefix .. "_unit:getLife", unit:getLife())
    log_value(prefix .. "_unit:getLife0", unit:getLife0())
    log_value(prefix .. "_unit:getFuel", unit:getFuel())

    local radarOn, radarTarget = unit:getRadar()
    log_value(prefix .. "_unit:getRadarRadarOn", radarOn)
    if radarTarget then
        log_value(prefix .. "_getPointRadarTarget", radarTarget:getPoint())
    else
        log_value(prefix .. "_unit:getRadarRadarTarget", "nil")
    end

    return unit
end

-- ---- logs mutual detection between the two test units via Controller ----
local function LogDetection(prefix, selfUnit, targetUnit)
    if not selfUnit or not targetUnit then return end

    local controller = selfUnit:getController()
    if not controller then
        log_value(prefix .. "_selfUnit:getController", "nil")
        return
    end

    -- 1) Direct check: "do I see that specific unit?"
    local detected, visible, lastTime, knownType, knownDistance, lastPos, lastVel =
        controller:isTargetDetected(targetUnit)

    log_value(prefix .. "_controller:isTargetDetectedDetected", detected)
    log_value(prefix .. "_controller:isTargetDetectedVisible", visible)
    log_value(prefix .. "_controller:isTargetDetectedLastTime", lastTime)
    log_value(prefix .. "_controller:isTargetDetectedKnownType", knownType)
    log_value(prefix .. "_controller:isTargetDetectedKnownDistance", knownDistance)
    log_value(prefix .. "_controller:isTargetDetectedLastPos", lastPos)
    log_value(prefix .. "_controller:isTargetDetectedLastVel", lastVel)

    -- 2) General sweep: "what does my radar picture look like right now?"
    local contacts = controller:getDetectedTargets()
    log_value(prefix .. "_controller:getDetectedTargetsContactCount", #contacts)
    for i, contact in ipairs(contacts) do
        local tag = prefix .. "_Contact" .. i
        log_value(tag .. "_Visible", contact.visible)
        log_value(tag .. "_TypeKnown", contact.type)
        log_value(tag .. "_DistanceKnown", contact.distance)
        if contact.object and contact.object:isExist() then
            log_value(tag .. "_Point", contact.object:getPoint())
        end
    end
end

-- ---- main test step, re-scheduled every LOG_PERIOD seconds ----
local function AIObsTestStep()
    log.write("AIObs", log.INFO, "=== AI DATA (t=" .. string.format("%.2f", timer.getTime()) .. ") ===")

    local blueUnit = LogUnitObs("Blue", BLUE_UNIT_NAME)
    local redUnit  = LogUnitObs("Red", RED_UNIT_NAME)

    LogDetection("BlueVsRed", blueUnit, redUnit)
    LogDetection("RedVsBlue", redUnit, blueUnit)

    return timer.getTime() + LOG_PERIOD
end

timer.scheduleFunction(AIObsTestStep, {}, timer.getTime() + 1)
log.write("AIObs", log.INFO, "=== AI DATA SCRIPT LOADED ===")
