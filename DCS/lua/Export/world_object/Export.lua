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
    -- Plan low-frequency event every 10 seconds
    LoCreateCoroutineActivity(1, 10.0, 10.0)
    Coroutines = {}
    Coroutines[1] = coroutine.create(LuaExportActivityNextEvent)
end

function LuaExportBeforeNextFrame()
    -- Empty - nothing to do before each frame
end

function LuaExportAfterNextFrame()
    -- Empty - no high frequency logging
end

-- =============================================================================
-- LOW FREQUENCY (every 10 seconds) - World Objects Only
-- =============================================================================
function LuaExportActivityNextEvent(t)
    log.write("Export", log.INFO, "=== WORLD OBJECTS (t=" .. string.format("%.2f", t) .. ") ===")
    
    -- Units (aircraft, vehicles, ships, etc.)
    local units = LoGetWorldObjects("units")
    if units then
        local count = 0
        for id, obj in pairs(units) do
            count = count + 1
            log_table("Unit_" .. id, obj)
        end
        log_kv("Units_Total", count)
    else
        log_str("Units", "nil")
    end
    
    -- Ballistic objects (bombs, rockets, shells in flight)
    local ballistic = LoGetWorldObjects("ballistic")
    if ballistic then
        local count = 0
        for id, obj in pairs(ballistic) do
            count = count + 1
            log_table("Ballistic_" .. id, obj)
        end
        log_kv("Ballistic_Total", count)
    else
        log_str("Ballistic", "nil")
    end
    
    -- Airdromes/airfields
    local airdromes = LoGetWorldObjects("airdromes")
    if airdromes then
        local count = 0
        for id, obj in pairs(airdromes) do
            count = count + 1
            log_table("Airdrome_" .. id, obj)
        end
        log_kv("Airdromes_Total", count)
    else
        log_str("Airdromes", "nil")
    end
    
    return t + 10.0
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