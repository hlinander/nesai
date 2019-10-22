function bcd_decode(n)
	return (bit.rshift(n, 4) * 10) + bit.band(n, 0x0f)
end

function brain_validate_frame(frame)
	local mode = read_cpu(0xC0)
	if 5 == mode then
		--
		-- No demo for you.
		--
		return false
	end
	if 4 == mode and 0 ~= read_cpu(0x58) then
		--
		-- In game mode and game over animation is running. We ded. 
		--
		return false
	end
	return frame < 10000
end

local last_input = 0

function brain_override_input(frame)
	if read_cpu(0xC0) < 4 then
		last_input = bit.bxor(last_input, 0x08)
		return last_input
	end
	return -1
end

local old_score = 0

function brain_get_reward(frame)
	if frame < 100 then
		--
		-- No score in demo mode
		--
		return 0
	end
	local score = bcd_decode(read_cpu(0x53))
		+ (bcd_decode(read_cpu(0x54)) * 100)
		+ (bcd_decode(read_cpu(0x55)) * 10000) + read_cpu(0xC0)
	--
	-- Normalize against a tetris on level 20
	--
	local reward = (score - old_score) 
	if reward > 10 then
		reward = 10
	end
	if reward < -10 then
		reward = -10
	end
	old_score = score
	return reward 
end
