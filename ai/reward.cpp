#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "model.h"
#include "reward.h"

std::vector<float> min_max_rewards(std::vector<float>& rewards)
{
	float min = *std::min_element(rewards.begin(), rewards.end());
	float max = *std::max_element(rewards.begin(), rewards.end());
	float absmax = std::max(fabs(min), fabs(max));
	std::vector<float> normalized_rewards(rewards);
	if(max - min > 0.00001)
	{
	// std::transform(normalized_rewards.begin(), normalized_rewards.end(), normalized_rewards.begin(),
	// 				[min, max](float r) -> float { return (r - min) / (max - min); });
	std::transform(normalized_rewards.begin(), normalized_rewards.end(), normalized_rewards.begin(),
					[absmax](float r) -> float { return r / absmax; });
	}
	return normalized_rewards;
}

std::vector<float> normalize_rewards(std::vector<float>& rewards) {
	std::vector<float> normalized_rewards(rewards);

	double sum = std::accumulate(rewards.begin(), rewards.end(), 0.0);
	double mean = sum / rewards.size();

	double sq_sum = std::inner_product(rewards.begin(), rewards.end(), rewards.begin(), 0.0);
	double stddev = std::sqrt(sq_sum / rewards.size() - mean * mean);

	if(stddev > 0.0f) {
		std::transform(normalized_rewards.begin(), normalized_rewards.end(), normalized_rewards.begin(),
					[mean, stddev](float r) -> float { return (r - mean) / stddev; });
	} 
	return normalized_rewards;
}

float calculate_rewards(Model &experience) {
	Reward ret;
	ret.rewards.resize(experience.get_frames());
	ret.adv.resize(experience.get_frames());
	std::fill(std::begin(ret.rewards), std::end(ret.rewards), 0.0f);
	std::fill(std::begin(ret.adv), std::end(ret.adv), 0.0f);
	ret.total_reward = 0.0;
	float reward = 0.0;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
        if(fabs(experience.immidiate_rewards[frame]) > 0.000000001) {
            // debug_log << "f " << frame << ": " << experience.immidiate_rewards[frame] << ", ";
        }
		reward += experience.immidiate_rewards[frame];
		reward *= 0.90;
		ret.rewards[frame] = reward;
		ret.adv[frame] = reward - experience.values[frame];
	}
	ret.total_reward = std::accumulate(ret.rewards.begin(), ret.rewards.end(), 0.0f);
	ret.adv = normalize_rewards(ret.adv);
	// ret.rewards = min_max_rewards(ret.rewards);
	experience.rewards = ret.rewards;
	experience.adv = ret.adv;
    // debug_log << "normed = [";
	// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
    //     debug_log << ret.rewards[frame] << ",";
    // }

	return ret.total_reward / static_cast<float>(experience.get_frames());
}