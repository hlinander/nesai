local old_level
local old_lives
local old_page
local old_screenx
local old_relx
local last_absolute_x
local idle_frames
local next_save_frame
local old_pstate
local nloads

local is_dead
local last_input

function brain_begin_rollout()
	old_level = 0
	old_lives = 0
	old_page = 0
	old_screenx = 0
	old_relx = 0
	last_absolute_x = 0
	is_dead = false
	last_input = 0
	idle_frames = 0
	next_save_frame = 0
	old_pstate = 0
	nloads = 0
end

function brain_validate_frame(frame)
	if is_dead then
		return false
	end
	if next_save_frame > 100 then
		save_state()
		next_save_frame = 0
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

	local pstate = read_cpu(0xE)
	local level = read_cpu(0x760)
	local page = read_cpu(0x71A)
	local screenx = read_cpu(0x71c)
	local relx = read_cpu(0x3ad)
	local speedx = read_int_cpu(0x57) / 0x28

	local absolute_x = page * 0x100 + screenx + relx
	local xscore = ((page - old_page) * 0x100) + (screenx - old_screenx) + (relx - old_relx)
	local reward = 0

	if screenx - old_screenx > 0 then
		next_save_frame = next_save_frame + 1
    end

	if 1 == read_cpu(0x770) and 3 == read_cpu(0x772) then
		reward = xscore + (math.abs(old_level - level) * 1000)
		--
		-- Penalize spazzing about like a fucking retard and not moving.
		-- Essentially, make a 10px movement every 2 seconds or lose 10 pts / frame
		--
		if math.abs(last_absolute_x - absolute_x) < 10 then
			idle_frames = idle_frames + 1
			if idle_frames > 120 then
				reward = reward - 1
				if idle_frames > 180 then
					-- is_dead = true
					if nloads < 5 then
						load_state()
						nloads = nloads + 1
					else
						is_dead = true
					end
					reward = reward - 100
					next_save_frame = 0
				end
			end
		else
			idle_frames = 0
			last_absolute_x = absolute_x
		end
	else
		last_absolute_x = absolute_x
		idle_frames = 0
	end

	if old_pstate ~= pstate and pstate == 0xb then
		if nloads < 5 then
			load_state()
			nloads = nloads + 1
		else
			is_dead = true
		end
		reward = reward - 100
		next_save_frame = 0
	end
	old_pstate = pstate

	-- if old_lives ~= lives then
	-- 	print("is dead")
	-- 	load_state()
	-- 	reward = reward - 100
	-- 	if 0 == lives then
	-- 		is_dead = true
	-- 	end
	-- 	old_lives = lives
	-- end

	-- if 0 ~= reward then
	-- 	print(reward)
	-- end

	old_page = page
	old_screenx = screenx
	old_relx = relx
	old_level = level
	-- print(speedx)
	return reward + speedx * 2
end


