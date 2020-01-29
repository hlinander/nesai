local last_input
local old_score
local is_dead
local nloads
local nmaxframes
local next_save_frame

function brain_begin_rollout()
	last_input = 0
	old_score = 0
	nloads = 0
	is_dead = false
	next_save_frame = 0
	nmaxframes = os.getenv('MAX_FRAMES')
	if nmaxframes == nil then
		nmaxframes = 3000
	else
		nmaxframes = tonumber(nmaxframes)
	end
end

function brain_validate_frame(frame)
	if is_dead then
		return false
	end
	if next_save_frame > 100 then
		save_state()
		next_save_frame = 0
	end
	return frame < nmaxframes
end

function brain_override_input(frame)
	if frame < 100 then
		last_input = last_input~8
		return last_input
	end
	return -1
end

function brain_get_reward(frame)
	if frame < 100 then
		return 0
	end
	local score = 1000000 * read_cpu(0x75) 
		+ 100000 * read_cpu(0x74) 
		+ 10000 * read_cpu(0x73) 
		+ 1000 * read_cpu(0x72)
		+ 100 * read_cpu(0x71)
		+ 10 * read_cpu(0x70)

	local died = (read_cpu(0x67) ~= 3)

	if died then
		if nloads < 5 then
			load_state()
			nloads = nloads + 1
		else
			is_dead = true
		end
		return -100
	end

	next_save_frame = next_save_frame + 1

	local reward = score - old_score
	old_score = score
	return reward
end


