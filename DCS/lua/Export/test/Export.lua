local DEVICE_ID    = 2
local COMMAND      = 3037
local ARG_THROTTLE = 757
local START_TIME   = 10   -- seconds after mission start
local HOLD_TIME    = 3    -- seconds to hold the button and the idle command

local startT = nil
local pressed, released = false, false

local function click(value)
    local ok, err = pcall(function()
        GetDevice(DEVICE_ID):performClickableAction(COMMAND, value)
    end)
    log.write("ThrottleOff", log.INFO, "click " .. value .. " ok=" .. tostring(ok) .. " err=" .. tostring(err))
end

function LuaExportBeforeNextFrame()
    local t = LoGetModelTime()
    if not t then return end

    -- 1) press the button once
    if not pressed and t > START_TIME then
        click(1)
        pressed, startT = true, t
    end

    -- 2) while holding: send throttle idle every frame
    if pressed and not released then
        LoSetCommand(2004, 1.0)

        -- 3) after HOLD_TIME: release the button and stop sending
        if t > startT + HOLD_TIME then
            click(0)
            released = true
        end
    end
end

function LuaExportAfterNextFrame()
    -- watch the result in dcs.log (search for ThrottleOff)
    local eng = LoGetEngineInfo()
    local rpm = eng and eng.RPM and eng.RPM.left
    local arg = GetDevice(0):get_argument_value(ARG_THROTTLE)
    log.write("ThrottleOff", log.INFO, "RPM=" .. tostring(rpm) .. "  button757=" .. tostring(arg))
end