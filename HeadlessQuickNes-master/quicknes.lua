-- An example script which allows playing via normal keyboard controls.
-- Use the arrow keys to move, Enter = Start, Backspace = Select,
-- Z is A, X is B and S and A are save and load state respectively.
-- The current measured FPS will be shown in the title bar.
hq = require("hqnes")

-- hq.gui.enable()      -- enable the gui, required if you want to do GUI stuff
-- hq.gui.setscale(2)   -- set the gui to scale to twice the regular size

hq.emu.loadrom(arg[1])
hq.emu.setframerate(0) -- limit the framerate to 60 fps (kinda sorta)

local frames = 0
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
    frames = frames + 1
    if 20000 == frames then
        print('.')
        frames = 0
    end
end
