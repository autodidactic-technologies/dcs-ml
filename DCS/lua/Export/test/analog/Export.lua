local test_phase = 0
local test_timer = 0
local test_duration = 5.0  

-- Needed analog commands (with value -1.0 to 1.0)
-- Thrust values are inverted for some internal reasons
local analog_commands = {
    {cmd = 2001, name = "Joystick pitch", value = 0.8},
    {cmd = 2002, name = "Joystick roll", value = -0.8},
    {cmd = 2003, name = "Joystick rudder", value = 0.8},
    {cmd = 2004, name = "Joystick thrust (both)", value = -1.0},
    {cmd = 2005, name = "Joystick left thrust", value = -1.0},
    {cmd = 2006, name = "Joystick right thrust", value = -1.0},
    {cmd = 2022, name = "Joystick trim pitch", value = 0.5},
    {cmd = 2023, name = "Joystick trim roll", value = -0.5},
    {cmd = 2024, name = "Trim rudder", value = 0.5},
}

function LuaExportStart()
    test_phase = 0
    test_timer = 0
end

function LuaExportBeforeNextFrame()
    test_timer = test_timer + 0.016
    
    if test_timer >= test_duration then
        test_timer = 0
        test_phase = test_phase + 1
    end
    
    if test_phase >= 1 and test_phase <= #analog_commands then
        local item = analog_commands[test_phase]
        LoSetCommand(item.cmd, item.value)
        
    elseif test_phase == #analog_commands + 1 then
        for _, item in ipairs(analog_commands) do
            LoSetCommand(item.cmd, 0.0)
        end
    end
end

function LuaExportAfterNextFrame()
end

function LuaExportStop()
end

function LuaExportActivityNextEvent(t)
    return t + 1.0
end