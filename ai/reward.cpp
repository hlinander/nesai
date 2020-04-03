#include "reward.h"
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "model.h"

std::vector<float> min_max_rewards(std::vector<float> &rewards) {
    float min = *std::min_element(rewards.begin(), rewards.end());
    float max = *std::max_element(rewards.begin(), rewards.end());
    float absmax = std::max(fabs(min), fabs(max));
    std::vector<float> normalized_rewards(rewards);
    if (max - min > 0.00001) {
        // std::transform(normalized_rewards.begin(), normalized_rewards.end(),
        // normalized_rewards.begin(), 				[min, max](float r) -> float { return (r - min) / (max -
        // min); });
        std::transform(normalized_rewards.begin(), normalized_rewards.end(),
                       normalized_rewards.begin(),
                       [absmax](float r) -> float { return r / absmax; });
    }
    return normalized_rewards;
}

std::vector<float> normalize_std(std::vector<float> &data) {
    std::vector<float> normalized_data(data);

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / data.size() - mean * mean);

    if (stddev > 0.0f) {
        std::transform(normalized_data.begin(), normalized_data.end(), normalized_data.begin(),
                       [mean, stddev](float r) -> float { return (r) / stddev; });
    }
    return normalized_data;
}

std::vector<float> normalize_mean_std(std::vector<float> &data) {
    std::vector<float> normalized_data(data);

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / (data.size()) - mean * mean);

    if (stddev > 0.0f) {
        std::transform(normalized_data.begin(), normalized_data.end(), normalized_data.begin(),
                       [mean, stddev](float r) -> float { return (r - mean) / stddev; });
    }
    return normalized_data;
}

float calculate_rewards(Model &experience, float discount) {
    experience.rewards.resize(experience.get_frames());
    experience.adv.resize(experience.get_frames());
    std::fill(std::begin(experience.rewards), std::end(experience.rewards), 0.0f);
    std::fill(std::begin(experience.adv), std::end(experience.adv), 0.0f);
    float reward = 0.0;
    for (int frame = experience.get_frames() - 1; frame >= 0; --frame) {
        reward *= discount;
        reward += experience.immidiate_rewards[frame];
        experience.rewards[frame] = reward;
        // ret.adv[frame] = reward;// - experience.values[frame];
    }
    experience.normalized_rewards = normalize_std(experience.rewards);
    // std::transform(experience.normalized_rewards.begin(), experience.normalized_rewards.end(),
    //                experience.values.begin(), experience.adv.begin(),
    //                [](float &reward, float &value) { return reward - value; });
    experience.adv = normalize_mean_std(experience.rewards);

    float total_reward = std::accumulate(experience.rewards.begin(),
                                         experience.rewards.end(), 0.0f);
    return total_reward / static_cast<float>(experience.get_frames());
}

float get_discount()
{
    const float default_discount = 0.995;
	const char *discount = getenv("DISCOUNT");
	if(discount)
	{
        try {
		    return std::stof(discount);
        }
        catch(const std::invalid_argument& ia)
        {
            std::cout << "INVALID LEARNING RATE" << std::endl;
            return default_discount;
        }
	}
	return default_discount;
}

