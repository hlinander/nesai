-- brain.lua
-- Load the brain library and adds ffi functions.

local core
local ffi = require "ffi"
local ok,hqn = pcall(ffi.load, "./libbrain.so")
if not ok then
    error("Failed to load brain library.")
end

local oldCpath = package.cpath
package.cpath = oldCpath .. ";" .. oldCpath:gsub("(%?)", "lib%1")
local brain = require "brain.luajit"
package.cpath = oldCpath

return brain
