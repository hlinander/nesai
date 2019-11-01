local old_level
local old_lives
local old_page
local old_screenx
local old_relx

local is_dead
local last_input

function brain_begin_rollout()
	old_level = 0
	old_lives = 0
	old_page = 0
	old_screenx = 0
	old_relx = 0
	is_dead = false
	last_input = 0
end

function brain_validate_frame(frame)
	if is_dead then
		return false
	end
	return frame < 10000
end

function brain_override_input(frame)
	if frame < 100 then
		last_input = last_input~8
		return last_input
	end
	return -1
end

function brain_get_reward(frame)
	local lives = read_cpu(0x75a)
	if frame < 100 then
		return 0
	elseif frame == 100 then
		old_lives = lives
	end

	local level = read_cpu(0x760)
	local page = read_cpu(0x71A)
	local screenx = read_cpu(0x71c)
	-- local relx = read_cpu(0x3ad)

	local xscore = ((page - old_page) * 0x100) + (screenx - old_screenx) -- + (relx - old_relx)
	local reward = 0

	if 1 == read_cpu(0x770) and 3 == read_cpu(0x772) then
		reward = xscore + (math.abs(old_level - level) * 1000)
	end

	if old_lives ~= lives then
		reward = reward - 1000
		if 0 == lives then
			is_dead = true
		end
		old_lives = lives
	end

	old_page = page
	old_screenx = screenx
	-- old_relx = relx
	old_level = level

	return reward
end


