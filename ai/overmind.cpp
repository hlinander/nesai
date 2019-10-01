#include <vector>
#include <unordered_map>
#include <algorithm>
#include "model.h"
#include "bm.h"

#define DEBUG(...)

const float LR = 0.0001;//0.00000001;
const int BATCH_SIZE = 1000;
const int PPO_EPOCHS = 5;

using stat_map = std::unordered_map<std::string, size_t>;
struct Reward {
	std::vector<float> rewards;
	float total_reward;
};

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
	DEBUG("Calculating rewards\n");
	Reward ret;
	ret.rewards.resize(experience.get_frames());
	std::fill(std::begin(ret.rewards), std::end(ret.rewards), 0.0f);
	ret.total_reward = 0.0;
	float reward = -1.0;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		reward += experience.immidiate_rewards[frame];
		reward *= 0.99;
		ret.rewards[frame] = reward;
	}
	ret.total_reward = std::accumulate(ret.rewards.begin(), ret.rewards.end(), 0.0f);
	ret.rewards = normalize_rewards(ret.rewards);
	return ret;
}

static std::string action_name(const ActionType &action)
{
    std::string s;
    for(size_t i = 0; i < ACTION_SIZE; ++i)
    {
        switch(static_cast<Action>(action[i]))
        {
            case Action::A: s.append("A"); break;
            case Action::B: s.append("B"); break;
            case Action::SELECT: s.append("SELECT"); break;
            case Action::START: s.append("START"); break;
            case Action::UP: s.append("UP"); break;
            case Action::DOWN: s.append("DOWN"); break;
            case Action::LEFT: s.append("LEFT"); break;
            case Action::RIGHT: s.append("RIGHT"); break;
            default: s.append("WAT"); break;
        }
        s.append("|");
    }
    if(!s.empty())
    {
        s.pop_back();
    }
    return s;
}

float update_model(Model &m, Model &experience, stat_map &stats, const float avg_reward, bool debug) {
	torch::Tensor loss = torch::tensor({0.0f});
	Reward reward = calculate_rewards(experience);

	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
		stats[action_name(experience.actions[frame])]++;
	}
	int steps = 0;
	DEBUG("Update with batches...\n");
	DEBUG("Frames: %d, Batchsize: %d\n", experience.get_frames(), BATCH_SIZE);
	// for (int frame = experience.get_frames() - 1; frame >= BATCH_SIZE; frame-=BATCH_SIZE) {
	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
		DEBUG("Frame %d\n", frame);
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);

		auto logp = m.forward_batch_nice(experience.get_batch(frame, frame + actual_bs));
		auto p = torch::sigmoid(logp);
		auto old_p = torch::sigmoid(experience.forward_batch_nice(frame, frame + actual_bs));

        std::array<float, BATCH_SIZE * ACTION_SIZE> rewards_batch{};
		for(int i = 0; i < actual_bs; ++i) {
            for(int j = 0; j < ACTION_SIZE; ++j) {
                if(experience.actions[frame + i][j] != 0) {
                    rewards_batch[i * ACTION_SIZE + j] = reward.rewards[frame + i];
                }
            }
		}
		auto trewards = torch::from_blob(static_cast<void*>(rewards_batch.data()), {actual_bs, ACTION_SIZE}, torch::kFloat32).to(m.net->device);
		auto masked_r = (p / old_p);
		auto lloss = torch::min(masked_r * trewards, torch::clamp(masked_r, 1.0 - 0.2, 1.0 + 0.2) * trewards).sum();
		(-lloss).backward();
	}
	DEBUG("Loss backwards\n");
	DEBUG("Returning...\n");
	return reward.total_reward;
}


int main(int argc, const char *argv[])
{
    if(argc < 2)
    {
        std::cout << "..." << std::endl;
        return 1;
    }

    if(0 == strcmp(argv[1], "create"))
    {
        if(argc < 3)
        {
            std::cout << "create <filename>" << std::endl;
            return 1;
        }

        Model m(LR);
        m.save_file(argv[2]);
    }
    else if(0 == strcmp(argv[1], "update"))
    {
        if(argc < 5)
        {
            std::cout << "update <model> <experiences> <model_out>" << std::endl;
            return 1;
        }

        Model m(LR);
        if(!m.load_file(argv[2]))
        {
            std::cout << "The horse has no carrot" << std::endl;
            return 1;
        }

        std::vector<Model> experiences;
        std::ifstream in{argv[3]};
        std::string str;
        {
            Benchmark b{"exp_load"};
            while(std::getline(in, str))
            {
                experiences.emplace_back(Model(LR));
                if(!experiences.back().load_file(str))
                {
                    std::cout << "Missing exp: " << str << std::endl;
                    return 3;
                }
            }
        }

        stat_map sm;
		for(int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
			Benchmark bepoch{"epoch"};
			m.optimizer.zero_grad();
            for(auto &e : experiences)
            {
                update_model(m, e, sm, 0.0, false);
            }
			m.optimizer.step();
        }
        m.save_file(argv[4]);
    }
    else
    {
        std::cout << "unknown command" << std::endl;
        return 1;
    }
    return 0;
}
