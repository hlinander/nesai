#pragma once

#include <vector>

struct Reward {
	std::vector<float> rewards;
	std::vector<float> adv;
	float total_reward;
};

struct Model;

std::vector<float> normalize_rewards(std::vector<float>& rewards);
float calculate_rewards(Model &experience, float discount);
float get_discount();