local test_phase = 0
local test_timer = 0
local test_duration = 3.0  -- Longer duration to observe effects

-- Group 1: Power/Engine Management
local power_commands = {
    {cmd = 309, name = "Engines start"},
    {cmd = 310, name = "Engines stop"},
    {cmd = 311, name = "Left engine start"},
    {cmd = 312, name = "Right engine start"},
    {cmd = 313, name = "Left engine stop"},
    {cmd = 314, name = "Right engine stop"},
    {cmd = 315, name = "Power on/off"},
    {cmd = 64, name = "Power up"},
    {cmd = 65, name = "Power down"},
    {cmd = 161, name = "Power up left engine"},
    {cmd = 162, name = "Power down left engine"},
    {cmd = 163, name = "Power up right engine"},
    {cmd = 164, name = "Power down right engine"},
}

-- Group 2: Flight Controls/Autopilot
local flight_controls = {
    {cmd = 62, name = "Autopilot"},
    {cmd = 63, name = "Auto-thrust"},
    {cmd = 427, name = "Plane autopilot override on"},
    {cmd = 428, name = "Plane autopilot override off"},
    {cmd = 429, name = "Plane route autopilot on/off"},
    {cmd = 59, name = "Altitude stabilization"},
    {cmd = 252, name = "Automatic spin recovery"},
    {cmd = 253, name = "Speed retention"},
    {cmd = 254, name = "Easy landing"},
    {cmd = 327, name = "Recover my plane"},
}

-- Group 3: Aircraft Mechanical
local mechanical = {
    {cmd = 68, name = "Gear toggle"},
    {cmd = 430, name = "Gear up"},
    {cmd = 431, name = "Gear down"},
    {cmd = 69, name = "Hook"},
    {cmd = 70, name = "Pack wings"},
    {cmd = 71, name = "Canopy"},
    {cmd = 72, name = "Flaps toggle"},
    {cmd = 145, name = "Flaps on"},
    {cmd = 146, name = "Flaps off"},
    {cmd = 73, name = "Air brake toggle"},
    {cmd = 147, name = "Air brake on"},
    {cmd = 148, name = "Air brake off"},
    {cmd = 74, name = "Wheel brakes on"},
    {cmd = 75, name = "Wheel brakes off"},
    {cmd = 76, name = "Release drogue chute"},
    {cmd = 155, name = "Refueling boom"},
    {cmd = 79, name = "Refuel on"},
    {cmd = 80, name = "Refuel off"},
}

-- Group 4: Weapons/Combat
local weapons = {
    {cmd = 283, name = "Switch master arm"},
    {cmd = 84, name = "Fire on"},
    {cmd = 85, name = "Fire off"},
    {cmd = 350, name = "Release weapon"},
    {cmd = 351, name = "Stop release weapon"},
    {cmd = 349, name = "Launch permission override"},
    {cmd = 81, name = "Salvo"},
    {cmd = 82, name = "Jettison weapons"},
    {cmd = 178, name = "Jettison fuel tanks"},
    {cmd = 113, name = "Cannon"},
    {cmd = 280, name = "Change cannon rate of fire"},
    {cmd = 281, name = "Change ripple quantity"},
    {cmd = 282, name = "Change ripple interval"},
    {cmd = 308, name = "Change ripple interval down"},
    {cmd = 284, name = "Change release mode"},
    {cmd = 357, name = "Drop flare once"},
    {cmd = 358, name = "Drop chaff once"},
    {cmd = 136, name = "Active jamming"},
    {cmd = 391, name = "Active IR jamming on/off"},
}

-- Group 5: Targeting/Modes
local targeting = {
    {cmd = 100, name = "Lock target"},
    {cmd = 143, name = "Unlock target"},
    {cmd = 102, name = "Change target"},
    {cmd = 101, name = "Change weapon"},
    {cmd = 106, name = "BVR mode"},
    {cmd = 107, name = "VS mode"},
    {cmd = 108, name = "Bore mode"},
    {cmd = 109, name = "Helmet mode"},
    {cmd = 110, name = "FI0 mode"},
    {cmd = 111, name = "A2G mode"},
    {cmd = 112, name = "Grid mode"},
    {cmd = 272, name = "Auto lock nearest aircraft"},
    {cmd = 273, name = "Auto lock center aircraft"},
    {cmd = 274, name = "Auto lock next aircraft"},
    {cmd = 275, name = "Auto lock previous aircraft"},
    {cmd = 276, name = "Auto lock nearest surface target"},
    {cmd = 277, name = "Auto lock center surface target"},
    {cmd = 278, name = "Auto lock next surface target"},
    {cmd = 279, name = "Auto lock previous surface target"},
    {cmd = 258, name = "Threat missile padlock"},
    {cmd = 259, name = "All missiles padlock"},
    {cmd = 184, name = "Padlock terrain point"},
}

-- Group 6: Radar/Sensors
local radar = {
    {cmd = 86, name = "Radar"},
    {cmd = 87, name = "EOS"},
    {cmd = 271, name = "Easy radar"},
    {cmd = 285, name = "Change radar RWS/TWS"},
    {cmd = 286, name = "Change RWR/SPO mode"},
    {cmd = 394, name = "Change radar PRF"},
    {cmd = 262, name = "Decrease radar scan area"},
    {cmd = 263, name = "Increase radar scan area"},
    {cmd = 392, name = "Laser range-finder on/off"},
    {cmd = 393, name = "Night TV on/off"},
    {cmd = 88, name = "Rotate radar antenna left"},
    {cmd = 89, name = "Rotate radar antenna right"},
    {cmd = 90, name = "Rotate radar antenna up"},
    {cmd = 91, name = "Rotate radar antenna down"},
    {cmd = 92, name = "Center radar antenna"},
    {cmd = 139, name = "Scan zone left"},
    {cmd = 140, name = "Scan zone right"},
    {cmd = 141, name = "Scan zone up"},
    {cmd = 142, name = "Scan zone down"},
    {cmd = 226, name = "Scan zone up right"},
    {cmd = 227, name = "Scan zone down right"},
    {cmd = 228, name = "Scan zone down left"},
    {cmd = 229, name = "Scan zone up left"},
    {cmd = 230, name = "Scan zone stop"},
    {cmd = 231, name = "Radar antenna up right"},
    {cmd = 232, name = "Radar antenna down right"},
    {cmd = 233, name = "Radar antenna down left"},
    {cmd = 234, name = "Radar antenna up left"},
    {cmd = 235, name = "Radar antenna stop"},
}

-- Group 7: Trimming
local trimming = {
    {cmd = 93, name = "Trim left"},
    {cmd = 94, name = "Trim right"},
    {cmd = 95, name = "Trim up"},
    {cmd = 96, name = "Trim down"},
    {cmd = 97, name = "Cancel trimming"},
    {cmd = 98, name = "Trim rudder left"},
    {cmd = 99, name = "Trim rudder right"},
    {cmd = 215, name = "Stop trimming"},
}

-- Group 8: Wingman/AWACS
local wingman = {
    {cmd = 114, name = "Wingman - RTB"},
    {cmd = 115, name = "Wingman - rejoin"},
    {cmd = 116, name = "Wingman - toggle formation"},
    {cmd = 117, name = "Wingman - join up"},
    {cmd = 118, name = "Wingman - attack my target"},
    {cmd = 119, name = "Wingman - cover my six"},
    {cmd = 267, name = "Ask AWACS home airbase"},
    {cmd = 268, name = "Ask AWACS available tanker"},
    {cmd = 269, name = "Ask AWACS nearest target"},
    {cmd = 270, name = "Ask AWACS declare target"},
}

-- Group 9: Cockpit/Displays
local cockpit = {
    {cmd = 122, name = "Sound on/off"},
    {cmd = 300, name = "Cockpit illumination"},
    {cmd = 316, name = "Altimeter pressure increase"},
    {cmd = 317, name = "Altimeter pressure decrease"},
    {cmd = 318, name = "Altimeter pressure stop"},
    {cmd = 245, name = "Coordinates units toggle"},
    {cmd = 328, name = "Toggle gear light Near/Far/Off"},
    {cmd = 409, name = "ThreatWarnSoundVolumeDown"},
    {cmd = 410, name = "ThreatWarnSoundVolumeUp"},
}

-- Group 10: Other/Misc
local misc = {
    {cmd = 52, name = "Suspend/resume model time"},
    {cmd = 53, name = "Accelerate model time"},
    {cmd = 54, name = "Step by step simulation"},
    {cmd = 246, name = "Disable time acceleration"},
    {cmd = 83, name = "Eject"},
    {cmd = 121, name = "Cobra"},
    {cmd = 120, name = "Take off from ship"},
    {cmd = 348, name = "Trains/cars toggle"},
    {cmd = 261, name = "Marker state"},
    {cmd = 264, name = "Marker state plane"},
    {cmd = 265, name = "Marker state rocket"},
    {cmd = 266, name = "Marker state plane ship"},
    {cmd = 176, name = "Drop snar once"},
    {cmd = 77, name = "Drop snar"},
    {cmd = 78, name = "Wingtip smoke"},
    {cmd = 186, name = "Plane up"},
    {cmd = 187, name = "Plane down"},
    {cmd = 188, name = "Bank left"},
    {cmd = 189, name = "Bank right"},
    {cmd = 193, name = "Nose down"},
    {cmd = 194, name = "Nose down end"},
    {cmd = 195, name = "Nose up"},
    {cmd = 196, name = "Nose up end"},
    {cmd = 197, name = "Bank left"},
    {cmd = 198, name = "Bank left end"},
    {cmd = 199, name = "Bank right"},
    {cmd = 200, name = "Bank right end"},
    {cmd = 201, name = "Rudder left"},
    {cmd = 202, name = "Rudder left end"},
    {cmd = 203, name = "Rudder right"},
    {cmd = 204, name = "Rudder right end"},
    {cmd = 386, name = "PlaneStabPitchBank"},
    {cmd = 387, name = "PlaneStabHbarBank"},
    {cmd = 388, name = "PlaneStabHorizont"},
    {cmd = 389, name = "PlaneStabHbar"},
    {cmd = 390, name = "PlaneStabHrad"},
    {cmd = 408, name = "PlaneStabCancel"},
    {cmd = 412, name = "PlaneIncreaseBase_Distance"},
    {cmd = 413, name = "PlaneDecreaseBase_Distance"},
    {cmd = 414, name = "PlaneStopBase_Distance"},
}

-- Combine all groups
local all_commands = {}
local function add_group(group, name)
    for _, item in ipairs(group) do
        table.insert(all_commands, item)
    end
end

-- CHOOSE WHICH GROUP TO TEST:
-- Comment/uncomment the group you want to test

--add_group(power_commands, "Power/Engine")
--add_group(flight_controls, "Flight Controls")
--add_group(mechanical, "Aircraft Mechanical")
add_group(weapons, "Weapons/Combat")
--add_group(targeting, "Targeting/Modes")
--add_group(radar, "Radar/Sensors")
--add_group(trimming, "Trimming")
--add_group(wingman, "Wingman/AWACS")
--add_group(cockpit, "Cockpit/Displays")
--add_group(misc, "Other/Misc")

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
    
    if test_phase >= 1 and test_phase <= #all_commands then
        local item = all_commands[test_phase]
        LoSetCommand(item.cmd)
        
    elseif test_phase == #all_commands + 1 then
        -- Reset common toggle states
        LoSetCommand(85)   -- Fire off
        LoSetCommand(351)  -- Stop release weapon
    end
end

function LuaExportAfterNextFrame()
end

function LuaExportStop()
end

function LuaExportActivityNextEvent(t)
    return t + 1.0
end