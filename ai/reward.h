#include <vector>
#include "model.h"

struct Reward {
	std::vector<float> rewards;
	float total_reward;
};

std::vector<float> normalize_rewards(std::vector<float>& rewards);
float calculate_rewards(Model &experience);