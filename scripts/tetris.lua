function brain_validate_frame(frame)
	return frame < 10000
end

function brain_get_reward()
	-- lines = read_cpu(0x50)
	return read_cpu(0x53) + read_cpu(0x54) * 100 + read_cpu(0x55) * 1000
end