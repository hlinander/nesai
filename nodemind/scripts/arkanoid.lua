function brain_validate_frame(frame)
	if read_cpu(0x000D) == 0 and frame > 500 then
		return false
    end
	return frame < 10000
end

local last_input = 0

function brain_override_input(frame)
	if frame < 100 then
		last_input = last_input~8
		return last_input
	end
	return -1
end

local old_score = 0

function brain_get_reward(frame)
	if frame < 10 then
		--
		-- No score in demo mode
		--
		return 0
	end
	local d6 = read_cpu(0x370)
	local d5 = read_cpu(0x371)
	local d4 = read_cpu(0x372)
	local d3 = read_cpu(0x373)
	local d2 = read_cpu(0x374)
	local d1 = read_cpu(0x375)
	local score = d6 * 100000 + d5 * 10000 + d4 * 1000 + d3 * 100 + d2 * 10 + d1

	local reward = (score - old_score) 
	old_score = score
	return reward 
end

