function brain_validate_frame(frame)
	return frame < 10000
end

local last_input = 0

function brain_override_input(frame)
	if frame < 100 then
		last_input = bit.bxor(last_input, 8)
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
	local score = (read_cpu(0x94) * 256 + read_cpu(0x90)) / 0x340
	--
	-- Normalize against a tetris on level 20
	--
	local reward = (score - old_score) 
	old_score = score
	return reward 
end


