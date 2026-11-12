-- F-16 steerpoint entry through the ICP / DED (STPT page)
-- Key sequence per waypoint:
--   RTN, 4(STPT), <stpt no>, ENTR,
--   DOWN x2 (skip MAN/AUTO), 2(N) <lat digits>, ENTR,
--   DOWN, 6(E) <lon digits>, ENTR,
--   DOWN, <elev digits>, ENTR,
--   UP x4, RTN

local UFC        = 17     -- device id
local START_TIME = 10     -- seconds after mission start (DED must be powered)
local PRESS_TIME = 0.15   -- seconds to hold a key
local GAP_TIME   = 0.15   -- seconds between keys

-- Lat: DDMM.mmm -> "DDMMmmm", Lon: DDDMM.mmm -> "DDDMMmmm", Elev: feet
local waypoints = {
    {stpt = 1, lat = "4552350",  lon = "03655971", elev = "20000"},
    {stpt = 2, lat = "4552450",  lon = "03755971", elev = "25000"},
}

local DIGIT = {
    ["0"] = 3002, ["1"] = 3003, ["2"] = 3004, ["3"] = 3005, ["4"] = 3006,
    ["5"] = 3007, ["6"] = 3008, ["7"] = 3009, ["8"] = 3010, ["9"] = 3011,
}
local ENTR     = {cmd = 3016, value = 1}
local DCS_UP   = {cmd = 3034, value = 1}
local DCS_DOWN = {cmd = 3035, value = -1}
local DCS_RTN  = {cmd = 3032, value = -1}

local keys = {}

local function add_key(key)
    table.insert(keys, key)
end

local function add_digits(text)
    for c in text:gmatch("%d") do
        add_key({cmd = DIGIT[c], value = 1})
    end
end

local function add_waypoint(wp)
    add_key(DCS_RTN)                         -- back to CNI page
    add_digits("4")                          -- STPT page
    add_digits(tostring(wp.stpt))
    add_key(ENTR)

    add_key(DCS_DOWN)                        -- MAN/AUTO (skip)
    add_key(DCS_DOWN)                        -- LAT
    add_digits("2")                          -- N
    add_digits(wp.lat)
    add_key(ENTR)

    add_key(DCS_DOWN)                        -- LNG
    add_digits("6")                          -- E
    add_digits(wp.lon)
    add_key(ENTR)

    add_key(DCS_DOWN)                        -- ELEV
    add_digits(wp.elev)
    add_key(ENTR)

    add_key(DCS_UP)                          -- asterisks back to STPT no
    add_key(DCS_UP)                          -- (the page remembers this
    add_key(DCS_UP)                          --  position next time)
    add_key(DCS_UP)
    add_key({log_stpt = wp.stpt})
    add_key(DCS_RTN)
end

add_key({log_route = "before"})
for _, wp in ipairs(waypoints) do
    add_waypoint(wp)
end
add_key({log_route = "after"})

local function click(cmd, value)
    local ok, err = pcall(function()
        GetDevice(UFC):performClickableAction(cmd, value)
    end)
    if not ok then
        log.write("Waypoints", log.ERROR, "cmd " .. cmd .. " failed: " .. tostring(err))
    end
end

local function log_ded(stpt)
    local ok, ded = pcall(list_indication, 6)
    log.write("Waypoints", log.INFO, "STPT " .. stpt .. " DED: " .. tostring(ok and ded or "n/a"))
end

-- Route as the export API sees it (obs data candidate)
local function log_route(tag)
    local ok, route = pcall(LoGetRoute)
    if not ok or not route or not route.route then
        log.write("Waypoints", log.INFO, "route " .. tag .. ": none")
        return
    end
    local next_num = route.goto_point and route.goto_point.next_point_num
    log.write("Waypoints", log.INFO, "route " .. tag .. ": " .. #route.route ..
        " points, goto=" .. tostring(next_num))
    for i, p in ipairs(route.route) do
        local wp = p.world_point
        local geo = LoCoordinatesToGeoCoordinates(wp.x, wp.z)
        log.write("Waypoints", log.INFO, string.format(
            "route %s #%d lat=%.5f lon=%.5f alt=%.0fm",
            tag, i, geo.latitude, geo.longitude, wp.y))
    end
end

local index = 1
local pressed = false
local nextT = nil

function LuaExportBeforeNextFrame()
    local t = LoGetModelTime()
    if not t or t < START_TIME or index > #keys then return end
    if nextT and t < nextT then return end

    local key = keys[index]

    if key.log_stpt or key.log_route then
        if key.log_stpt then log_ded(key.log_stpt) else log_route(key.log_route) end
        index = index + 1
        nextT = t + GAP_TIME
    elseif not pressed then
        click(key.cmd, key.value)
        pressed = true
        nextT = t + PRESS_TIME
    else
        click(key.cmd, 0)
        pressed = false
        index = index + 1
        nextT = t + GAP_TIME
        if index > #keys then
            log.write("Waypoints", log.INFO, "all waypoints entered")
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
