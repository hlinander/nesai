function brain_begin_rollout()
end

function brain_validate_frame(frame)
	return frame < 1000
end

function brain_override_input(frame)
	return -1
end

function brain_get_reward(frame)
    return 0
end