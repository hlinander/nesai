#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "model.h"
#include "reward.h"

std::vector<float> normalize_rewards(std::vector<float>& rewards) {
	std::vector<float> normalized_rewards(rewards);

	double sum = std::accumulate(rewards.begin(), rewards.end(), 0.0);
	double mean = sum / rewards.size();

	double sq_sum = std::inner_product(rewards.begin(), rewards.end(), rewards.begin(), 0.0);
	double stddev = std::sqrt(sq_sum / rewards.size() - mean * mean);
	
	if(stddev > 0.0f) {
		std::transform(normalized_rewards.begin(), normalized_rewards.end(), normalized_rewards.begin(),
					[mean, stddev](float r) -> float { return (r) / stddev; });
	} 
	return normalized_rewards;
}

Reward calculate_rewards(Model &experience) {
	Reward ret;
	ret.rewards.resize(experience.get_frames());
	std::fill(std::begin(ret.rewards), std::end(ret.rewards), 0.0f);
	ret.total_reward = 0.0;
	float reward = 0.0;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
        if(fabs(experience.immidiate_rewards[frame]) > 0.000000001) {
            // debug_log << "f " << frame << ": " << experience.immidiate_rewards[frame] << ", ";
        }
		reward += experience.immidiate_rewards[frame];
		reward *= 0.8;
		ret.rewards[frame] = reward;
	}
	ret.total_reward = std::accumulate(ret.rewards.begin(), ret.rewards.end(), 0.0f);
	// ret.rewards = normalize_rewards(ret.rewards);
    // debug_log << "normed = [";
	// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
    //     debug_log << ret.rewards[frame] << ",";
    // }

	return ret;
}