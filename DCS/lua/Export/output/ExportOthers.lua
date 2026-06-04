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

-- Helper: log string value
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

local lastExportTime = 0

function LuaExportStart()
    log.write("Export", log.INFO, "=== Export started ===")
end

function LuaExportBeforeNextFrame()
    -- Empty
end

function LuaExportAfterNextFrame()
    local t = LoGetModelTime()
    
    if t - lastExportTime >= 1.0 then
        lastExportTime = t
        
        log.write("Export", log.INFO, "=== EXPORT DATA (t=" .. string.format("%.2f", t) .. ") ===")
        
        -- TWS (Threat Warning System)
        local tws = LoGetTWSInfo()
        if tws then
            log_table("TWS", tws)
        else
            log_str("TWS", "nil")
        end

        -- Target Information (all targets)
        local targets = LoGetTargetInformation()
        if targets then
            log_kv("AllTargets_Count", #targets)
            for i, trg in ipairs(targets) do
                log_table("AllTarget_"..i, trg)
            end
        else
            log_str("AllTargets", "nil")
        end

        -- Locked Target Information
        local locked = LoGetLockedTargetInformation()
        if locked then
            log_kv("LockedTargets_Count", #locked)
            for i, trg in ipairs(locked) do
                log_table("LockedTarget_"..i, trg)
            end
        else
            log_str("LockedTargets", "nil")
        end

        -- Wingmen Info
        local wing = LoGetWingInfo()
        if wing then
            log_kv("Wingmen_Count", #wing)
            for i, w in ipairs(wing) do
                log_table("Wingman_"..i, w)
            end
        else
            log_str("WingInfo", "nil")
        end

        -- Wing Targets
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
    end
end

function LuaExportStop()
    log.write("Export", log.INFO, "=== Export stopped ===")
end