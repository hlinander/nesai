#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <experimental/filesystem>
#include "model.h"
#include "bm.h"
#include "json.hpp"

#define DEBUG(...)

static int get_batch_size()
{
	const char *bs = getenv("BATCH_SIZE");
	if(bs)
	{
		return atoi(bs);
	}
	return 512;
}

static int get_ppo_epochs()
{
	const char *bs = getenv("PPO_EPOCHS");
	if(bs)
	{
		return atoi(bs);
	}
	return 3;
}

const float LR = 0.001;//0.00000001;
static int BATCH_SIZE = get_batch_size();
const int PPO_EPOCHS = get_ppo_epochs();
const bool DEBUG = nullptr != getenv("DEBUG");

static std::ofstream debug_log;
static std::ofstream doom_log;
static nlohmann::json json;

using stat_map = std::unordered_map<std::string, size_t>;
struct Reward {
	std::vector<float> rewards;
	float total_reward;
};

void print_stats(const stat_map &s, size_t total_frames) {
	if(total_frames) {
		for(auto it = s.begin(); s.end() != it; ++it) {
			std::cout << (it->first) << ": " << static_cast<int>(100.0 * static_cast<float>(it->second) / total_frames) << "%, ";
		}
		std::cout << std::endl;
	}
}

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
	float reward = 0.0;
    debug_log << "LUA rewards " << std::endl;
	for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
        if(fabs(experience.immidiate_rewards[frame]) > 0.000000001) {
            debug_log << "f " << frame << ": " << experience.immidiate_rewards[frame] << ", ";
        }
		reward += experience.immidiate_rewards[frame];
		reward *= 0.95;
		ret.rewards[frame] = reward;
	}
	ret.total_reward = std::accumulate(ret.rewards.begin(), ret.rewards.end(), 0.0f);
	ret.rewards = normalize_rewards(ret.rewards);
    // debug_log << "normed = [";
	// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
    //     debug_log << ret.rewards[frame] << ",";
    // }

	return ret;
}

static std::string action_name(const ActionType &action)
{
    std::string s;
    for(size_t i = 0; i < ACTION_SIZE; ++i)
    {
        if(action[i])
        {
            switch(static_cast<Action>(i))
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
    }
    if(!s.empty())
    {
        s.pop_back();
    }
    return s;
}

void analyze_step(Model &after, Model &experience) {
	Reward reward = calculate_rewards(experience);
    for(size_t frame = 0; frame < (size_t)experience.get_frames(); ++frame)
    {
        auto prob_before = torch::sigmoid(experience.forward(experience.states[frame]));
        auto prob_after = torch::sigmoid(after.forward(experience.states[frame]));
        debug_log << "Reward: " << reward.rewards[frame] << std::endl;
        debug_log << "Before: " << prob_before << std::endl;
        debug_log << "After: " << prob_after << std::endl;
    }
}

float update_model(Model &m, Model &experience, stat_map &stats, const float avg_reward, bool debug) {
	torch::Tensor loss = torch::tensor({0.0f});
	Reward reward = calculate_rewards(experience);

	// for (int frame = experience.get_frames() - 1; frame >= 1; --frame) {
	// 	stats[action_name(experience.actions[frame])]++;
	// }

	// std::cout << "Frames: " << experience.get_frames() << ", Batchsize: " << BATCH_SIZE << std::endl;

	std::vector<float> rewards_batch;
	rewards_batch.resize(BATCH_SIZE);
	std::vector<float> action_batch;
	action_batch.resize(BATCH_SIZE * ACTION_SIZE);

	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
		// std::cout << "Frame: " << frame << ", Batchsize: " << actual_bs << std::endl;
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);
        if(actual_bs == 1) {
            break;
        }
        // auto thresh = (ACTION_THRESHOLD * torch::ones({(long)actual_bs, ACTION_SIZE})).to(m.net->device);
        auto v = m.value_net->forward(experience.get_batch(frame, frame + actual_bs));
		auto logp = m.forward_batch_nice(experience.get_batch(frame, frame + actual_bs));
		auto p = torch::sigmoid(logp);
		auto old_p = torch::sigmoid(experience.forward_batch_nice(frame, frame + actual_bs));
		// auto p = logp;
		// auto old_p = experience.forward_batch_nice(frame, frame + actual_bs);


        // std::fill(std::begin(pi_batch), std::end(pi_batch), 0.0f);
		for(size_t i = 0; i < actual_bs; ++i) {
            rewards_batch[i] = reward.rewards[frame + i];
            for(size_t j = 0; j < ACTION_SIZE; ++j) {
                action_batch[i * ACTION_SIZE + j] = experience.actions[frame + i][j];
                // if(experience.actions[frame + i][j] != 0) {
                //     rewards_batch[i * ACTION_SIZE + j] = reward.rewards[frame + i];
                //     // debug_log << reward.rewards[frame + i] << std::endl;
                // }
            }
		}
        /*
            a = (a_1, a_2, a_3, ...., a_n), a_i = {0, 1}

            a = (1, 1, 0, 0, 1)

            pi(a | s) = \prod_i [ a_i phi_i(s) + (1 - a_i)(1 - phi_i(s)) ]

            LH = \sum_b [ pi(a_b | s_b) / pi_old(a_b | s_b) * R ]

        */
		auto trewards = torch::from_blob(static_cast<void*>(rewards_batch.data()), {(long)actual_bs, 1}, torch::kFloat32);
        auto trewards_gpu = trewards.to(m.net->device);
        auto trewards_minus_V = trewards_gpu;// - torch::clamp(v, 0.0f, 10000.0f);
		auto torch_actions = torch::from_blob(static_cast<void*>(action_batch.data()), {(long)actual_bs, ACTION_SIZE}, torch::kFloat32);
        auto gpu_actions = torch_actions.to(m.net->device);

        auto pi = gpu_actions * p + (1.0f - gpu_actions) * (1.0f - p);
        auto old_pi = gpu_actions * old_p + (1.0f - gpu_actions) * (1.0f - old_p);
        auto prod_pi = pi.prod(1);
        auto prod_old_pi = old_pi.prod(1);
        // debug_log << trewards << std::endl;

        //.to(m.net->device);
		// auto masked_r = torch::exp(p - old_p);
        auto r = prod_pi / torch::clamp(prod_old_pi, 0.0001f, 1.0f);
		auto lloss = torch::min(r * trewards_gpu, torch::clamp(r, 1.0 - 0.2, 1.0 + 0.2) * trewards_gpu);
		auto sloss = lloss.mean();

		(-sloss).backward();

        m.value_optimizer.zero_grad();
        auto vloss = ((trewards_gpu - v) * (trewards_gpu - v)).sum();
        (vloss).backward();
        m.value_optimizer.step();
	}

	DEBUG("Loss backwards\n");
	DEBUG("Returning...\n");
	return reward.total_reward;
}


int main(int argc, const char *argv[])
{
    srand(time(0));
    if(DEBUG)
    {
        debug_log.open("overmind.log");
    }
    debug_log.open("doom.log", std::ofstream::app);

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
    else if(0 == strcmp(argv[1], "load")) 
    {
        Model m(LR);
        if(!m.load_file(argv[2]))
        {
            std::cout << "The horse has no carrot" << std::endl;
            return 1;
        }
    }
    else if(0 == strcmp(argv[1], "update_stdin"))
    {
        std::string line;
        while(!std::getline(std::cin, line))
        {

        }
    }
    else if(0 == strcmp(argv[1], "update"))
    {
        if(argc < 6)
        {
            std::cout << "update <model> <experiences> <model_out> <generation>" << std::endl;
            return 1;
        }
        constexpr size_t arg_model = 2;
        constexpr size_t arg_experiences = 3;
        constexpr size_t arg_model_out = 4;
        constexpr size_t arg_generation = 5;
        std::cout << "Update!" << std::endl;
        Benchmark full_ud("full_update");
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
            std::cout << "Reading lines..." << std::endl;
            while(std::getline(in, str))
            {
                std::cout << str << std::endl;
                try {
                    experiences.emplace_back(Model(LR));
                    if(!experiences.back().load_file(str))
                    {
                        std::cout << "Missing exp: " << str << std::endl;
                        return 3;
                    }
                    std::cout << "Loaded experience with " << experiences.back().get_frames() << " frames" << std::endl;
                } catch(const std::exception &e) {
                    std::cout << "FIXTHISFFS:" << e.what() << std::endl;
                }
            }
        }

        stat_map sm;
        int total_frames = 0;
        float reward = 0;
        int n_rewards = 0;
        std::cout << "Starting updates" << std::endl;
		for(int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
			Benchmark bepoch{"epoch"};
			m.optimizer.zero_grad();
            m.value_optimizer.zero_grad();
            for(auto &e : experiences)
            {
                reward += update_model(m, e, sm, 0.0, false);
                ++n_rewards;
                total_frames += e.get_frames();
            }
			m.optimizer.step();
			m.value_optimizer.step();
            if(m.net->isnan()) {
                std::cout << "Doomed tensors!" << std::endl;
                for(auto &e : experiences)
                {
                    std::cout << "experience" << std::endl;
                    for(int i = 0; i < e.get_frames(); ++i)
                    {
                        if(fabs(e.immidiate_rewards[i]) > 0.0001)
                            doom_log << e.immidiate_rewards[i] << std::endl;
                    }
                }

            }
        }
#ifndef NO_HAMPUS
        {
            Benchmark hampe_dbg("hampe_dbg");
            auto np = m.net->named_parameters();
            auto oldp = experiences[0].net->named_parameters();
            json["parameter_stats"] = nlohmann::json({});
            json["parameters"] = nlohmann::json({});
            json["dparameters"] = nlohmann::json({});
            json["rewards"] = calculate_rewards(experiences[0]).rewards;
            json["actions"] = experiences[0].actions;
            for(auto &ref : np.pairs()) {
                std::cout << ref.first << std::endl;
                std::cout << "mean: " << ref.second.mean().item<float>() << " std: " << ref.second.std().item<float>() << std::endl;
                auto cp = ref.second.to(torch::kCPU);
                auto dcp = (ref.second - oldp[ref.first]).to(torch::kCPU);

                int64_t nel = std::min(cp.numel(), static_cast<int64_t>(1000));
                std::vector<float> p(cp.data<float>(), cp.data<float>() + nel);
                std::vector<float> dp(dcp.data<float>(), dcp.data<float>() + nel);
                json["parameters"][ref.first]["values"] = p;
                json["dparameters"][ref.first]["values"] = dp;
                json["parameter_stats"][ref.first]["mean"] = ref.second.mean().item<float>();
                json["parameter_stats"][ref.first]["stddev"] = ref.second.std().item<float>();
            }
            json["mean_reward"] = reward / static_cast<float>(n_rewards);
            std::vector<float> values;
            values.resize(experiences[0].get_frames());
            for (int frame = 0; frame < experiences[0].get_frames(); frame+=BATCH_SIZE) {
                size_t actual_bs = std::min(BATCH_SIZE, experiences[0].get_frames() - frame);
                if(actual_bs == 1) {
                    break;
                }
                auto v = m.value_net->forward(experiences[0].get_batch(frame, frame + actual_bs)).to(torch::kCPU);
                std::copy(v.data<float>(), v.data<float>() + actual_bs, values.begin() + frame);
            }
            json["values"] = values;
        }
#endif
        {
            Benchmark save("savefile");
            m.save_file(argv[4]);
        }
#ifndef NO_HAMPUS
        {
            Benchmark hampe2("hampe2");
            std::experimental::filesystem::create_directories("metrics/");
            std::ofstream out(std::string("metrics/") + argv[arg_generation] + ".json");
            out << std::setw(4) << json << std::endl;
        }
#endif
    }
    else
    {
        std::cout << "unknown command" << std::endl;
        return 1;
    }
    return 0;
}
