-- An example script which allows playing via normal keyboard controls.
-- Use the arrow keys to move, Enter = Start, Backspace = Select,
-- Z is A, X is B and S and A are save and load state respectively.
-- The current measured FPS will be shown in the title bar.
hq = require("hqnes")
brain = require("brain")

function bool(value)
  if value ~= 0 then
    return true
  end
  return false
end

function run_brain_mode()
    while brain.on_frame() do
        local bits = brain.controller_bits()
        hq.joypad.set{
            right = bool(bit.band(bits, 0x80)),
            left = bool(bit.band(bits, 0x40)),
            down = bool(bit.band(bits, 0x20)),
            up = bool(bit.band(bits, 0x10)),
            start = bool(bit.band(bits, 0x08)),
            select = bool(bit.band(bits, 0x04)),
            b = bool(bit.band(bits, 0x02)),
            a = bool(bit.band(bits, 0x01)),
        }
        hq.emu.frameadvance()
    end
end

function run_human_mode()
    local kbprev = {}
    while true do
        local kb = hq.input.get()  -- get the current keyboard state
        hq.joypad.set{
            left = kb.Left or false,
            right = kb.Right or false,
            up = kb.Up or false,
            down = kb.Down or false,
            start = kb.Return or false,
            select = kb.Backspace or false,
            a = kb.Z or false,
            b = kb.X or false,
        }
        if kb.S and not kbprev.S then
            hq.savestate.save("1.savestate")
        end
        if kb.A and not kbprev.A then
            hq.savestate.load("1.savestate")
        end
        if kb.F11 and not kbprev.F11 then
            hq.gui.setfullscreen(not gui.isfullscreen())
        end
        -- hq.gui.settitle("Headless Quick NES " .. tostring(hq.emu.getfps()))
        -- print(hq.emu.getfps())
        hq.emu.frameadvance() -- advance the emulator
        kbprev = kb -- set prev keyboard state to current
    end
end

if not brain.headless() then
    hq.gui.enable()      -- enable the gui, required if you want to do GUI stuff
    hq.gui.setscale(2)   -- set the gui to scale to twice the regular size
    hq.emu.setframerate(0)
else
    hq.emu.setframerate(0)
end

hq.emu.loadrom(arg[1])

if brain.enabled() then
    run_brain_mode()
else
    run_human_mode()
end