local last_input
local old_score
local old_lives
local nloads
local nmaxframes
local next_save_frame

function brain_begin_rollout()
	last_input = 0
	old_score = 0
	old_lives = 3
	nloads = 0
	next_save_frame = 0
	nmaxframes = os.getenv('MAX_FRAMES')
	if nmaxframes == nil then
		nmaxframes = 3000
	else
		nmaxframes = tonumber(nmaxframes)
	end
end

function brain_validate_frame(frame)
	-- is_dying && no lifes left
	if 2 == read_cpu(0x100) and 0 == read_cpu(0x20) then
		return false
	end
	if next_save_frame > 200 then
		save_state()
		next_save_frame = 0
	end
	return frame < nmaxframes
end

function brain_override_input(frame)
	if frame < 200 then
		last_input = last_input~8
		return last_input
	end
	return -1
end

function brain_get_reward(frame)
	if frame < 200 then
		return 0
	end

	local score = ((read_cpu(0x7e6) & 0xF0) >> 0x10) * 1000000
		+ ((read_cpu(0x7e6) & 0x0F) * 100000)
		+ ((read_cpu(0x7e5) & 0xF0) >> 0x10) * 10000
		+ ((read_cpu(0x7e5) & 0x0F) * 1000)
		+ ((read_cpu(0x7e4) & 0xF0) >> 0x10) * 100
		+ ((read_cpu(0x7e4) & 0x0F) * 10)

	local reward = score - old_score
	old_score = score

	local lives = read_cpu(0x20)
	if lives < old_lives then
		if nloads < 5 then
			load_state()
			nloads = nloads + 1
		end
		reward = -1000
	end
	old_lives = lives
	-- if alive...
	if 1 == read_cpu(0x100) then
		next_save_frame = next_save_frame + 1
	end
	return reward
end


