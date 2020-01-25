#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <experimental/filesystem>
#include <stdexcept>
#include "model.h"
#include "bm.h"
#include "json.hpp"
#include "replay.h"
#include "reward.h"
#include "rds.hpp"

#define DEBUG(...)

static int get_batch_size()
{
	const char *bs = getenv("BATCH_SIZE");
	if(bs)
	{
		return atoi(bs);
	}
	return 1024;
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

static float get_learning_rate()
{
    const float default_lr = 0.001;
	const char *lr = getenv("LR");
	if(lr)
	{
        try {
		    return std::stof(lr);
        }
        catch(const std::invalid_argument& ia)
        {
            std::cout << "INVALID LEARNING RATE" << std::endl;
            return default_lr;
        }
	}
	return default_lr;
}

const float LR = get_learning_rate();
static int BATCH_SIZE = get_batch_size();
const int PPO_EPOCHS = get_ppo_epochs();
const bool DEBUG = nullptr != getenv("DEBUG");

static std::ofstream debug_log;
static std::ofstream doom_log;
static nlohmann::json json;
static rds_data rds;

using stat_map = std::unordered_map<std::string, size_t>;

void print_stats(const stat_map &s, size_t total_frames) {
	if(total_frames) {
		for(auto it = s.begin(); s.end() != it; ++it) {
			std::cout << (it->first) << ": " << static_cast<int>(100.0 * static_cast<float>(it->second) / total_frames) << "%, ";
		}
		std::cout << std::endl;
	}
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
	// Reward reward = calculate_rewards(experience);
    after.net->eval();
    experience.net->eval();
    for(size_t frame = 0; frame < (size_t)experience.get_frames(); ++frame)
    {
        auto prob_before = torch::softmax(experience.forward(experience.states[frame]), 1).to(torch::kCPU);
        auto prob_after = torch::softmax(after.forward(experience.states[frame]), 1).to(torch::kCPU);
        auto pb = prob_before.accessor<float, 2>();
        auto pa = prob_after.accessor<float, 2>();

        debug_log << "Reward: " << experience.rewards[frame] << " ";
        for(int i = 0; i < prob_before.sizes()[1]; ++i)
        {
            if(experience.actions[frame][i] == 1)
            {
                debug_log << " **[" << pb[0][i] << " -> " << pa[0][i] << "]**, ";
            }
            else
            {
                // debug_log << pb[0][i] << " -> " << pa[0][i] << ", ";
            }
        }
        debug_log << std::endl;
        // debug_log << "Reward: " << experience.rewards[frame] << std::endl;
        // debug_log << "Before: " << prob_before << std::endl;
        // debug_log << "After: " << prob_after << std::endl;
    }
    debug_log << std::endl << std::flush;
}

void distill(Model &exp)
{
	float min = *std::min_element(exp.rewards.begin(), exp.rewards.end());
	float max = *std::max_element(exp.rewards.begin(), exp.rewards.end());
	float absmax = std::max(fabs(min), fabs(max));

    size_t active_frames = 0;
    for(int i=exp.get_frames() - 1; i >= 0; --i)
    {
        if(fabs(exp.rewards[i]) < absmax * 0.1)
        {
            //exp.remove_frame(i);
            exp.rewards[i] = 0.0f;
        }
        else
        {
            active_frames++;
        }
    }
    std::cout << "ACTIVE FRAMES: " << active_frames << std::endl;
}

void update_model_softmax(Model &m, Model &experience, stat_map &stats, const float avg_reward, bool debug) {
	torch::Tensor loss = torch::tensor({0.0f});
	// Reward reward = calculate_rewards(experience);

	std::vector<float> rewards_batch;
	rewards_batch.resize(BATCH_SIZE);
	std::vector<float> action_batch;
	action_batch.resize(BATCH_SIZE * ACTION_SIZE);
    std::vector<long> action_indices;
	action_indices.resize(BATCH_SIZE);

    m.net->train();

	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);
        if(actual_bs == 1) {
            break;
        }
        auto v = m.value_net->forward(experience.get_batch(frame, frame + actual_bs));
		auto logp = m.forward_batch_nice(experience.get_batch(frame, frame + actual_bs));
		// auto p = torch::softmax(logp, 1);
		auto old_logp = experience.forward_batch_nice(frame, frame + actual_bs);

		for(size_t i = 0; i < actual_bs; ++i) {
            rewards_batch[i] = experience.rewards[frame + i];
            for(size_t j = 0; j < ACTION_SIZE; ++j) {
                action_batch[i * ACTION_SIZE + j] = experience.actions[frame + i][j];
                if(experience.actions[frame + i][j] == 1)
                {
                    action_indices[i] = static_cast<long>(j);
                }
            }
		}

		auto trewards = torch::from_blob(static_cast<void*>(rewards_batch.data()), {(long)actual_bs, 1}, torch::kFloat32);
        auto trewards_gpu = trewards.to(m.net->device);
        auto trewards_minus_V = trewards_gpu;// - v.detach(); //torch::clamp(v, 0.0f, 10000.0f);
		auto torch_actions = torch::from_blob(static_cast<void*>(action_batch.data()), {(long)actual_bs, ACTION_SIZE}, torch::kFloat32);
		auto torch_action_indices = torch::from_blob(static_cast<void*>(action_indices.data()), {(long)actual_bs, 1}, torch::kLong);
        auto gpu_actions = torch_actions.to(m.net->device);
        auto gpu_action_indices = torch_action_indices.to(m.net->device);
        // std::cout << torch_actions << std::endl;
        // std::cout << "p: ";
        // auto aa = torch_action_indices.accessor<long, 2>();
        // for(int i = 0; i < actual_bs; ++i)
        // {
        //     std::cout << std::endl << "old_logp: (";
        //     for(int j = 0; j < ACTION_SIZE; ++j)
        //     {
        //         std::cout << old_logp[i][j] << ", ";
        //     }
        //     std::cout << ") : " << aa[i][0] << std::endl;
        // }
        // std::cout << std::endl << "g: ";
        // for(auto x: gpu_actions.sizes())
        //     std::cout << x << ", ";
        // std::cout << std::flush;
        // auto r = p * gpu_actions / torch::clamp(old_p, 0.0001f, 1.0f);
        auto logp_action = logp.gather(1, gpu_action_indices);
        auto old_logp_action = old_logp.gather(1, gpu_action_indices);
        auto r = torch::exp(logp_action - old_logp_action.detach());
        // std::cout << (r * trewards_gpu).mean().item<float>() << std::endl;
		auto lloss = torch::min(r * trewards_gpu, torch::clamp(r, 1.0 - 0.1, 1.0 + 0.1) * trewards_gpu);
	    auto sloss = lloss.mean();

		(-sloss).backward();

        m.value_optimizer.zero_grad();
        auto vloss = ((trewards_gpu - v) * (trewards_gpu - v)).sum();
        (vloss).backward();
        m.value_optimizer.step();
	}

	DEBUG("Loss backwards\n");
	DEBUG("Returning...\n");
}

void update_model(Model &m, Model &experience, stat_map &stats, const float avg_reward, bool debug) {
	torch::Tensor loss = torch::tensor({0.0f});
	// Reward reward = calculate_rewards(experience);

	std::vector<float> rewards_batch;
	rewards_batch.resize(BATCH_SIZE);
	std::vector<float> action_batch;
	action_batch.resize(BATCH_SIZE * ACTION_SIZE);
    m.net->train();

	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);
        if(actual_bs == 1) {
            break;
        }
        auto v = m.value_net->forward(experience.get_batch(frame, frame + actual_bs));
		auto logp = m.forward_batch_nice(experience.get_batch(frame, frame + actual_bs));
		auto p = torch::sigmoid(logp);
		auto old_p = torch::sigmoid(experience.forward_batch_nice(frame, frame + actual_bs));

		for(size_t i = 0; i < actual_bs; ++i) {
            rewards_batch[i] = experience.rewards[frame + i];
            for(size_t j = 0; j < ACTION_SIZE; ++j) {
                action_batch[i * ACTION_SIZE + j] = experience.actions[frame + i][j];
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
        auto trewards_minus_V = trewards_gpu - v; //torch::clamp(v, 0.0f, 10000.0f);
		auto torch_actions = torch::from_blob(static_cast<void*>(action_batch.data()), {(long)actual_bs, ACTION_SIZE}, torch::kFloat32);
        auto gpu_actions = torch_actions.to(m.net->device);

        auto pi = gpu_actions * p + (1.0f - gpu_actions) * (1.0f - p);
        auto old_pi = gpu_actions * old_p + (1.0f - gpu_actions) * (1.0f - old_p);
        auto prod_pi = pi.prod(1);
        auto prod_old_pi = old_pi.prod(1);

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
	// return reward.total_reward;
}


int main(int argc, const char *argv[])
{
    srand(time(0));
    // if(DEBUG)
    // {
    // }
    // debug_log.open("doom.log", std::ofstream::app);

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
    else if(0 == strcmp(argv[1], "replay"))
    {
        Model m(LR);
        if(!m.load_file(argv[2]))
        {
            std::cout << "The horse has no carrot" << std::endl;
            return 1;
        }
        replay(m);
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
        if(argc < 7)
        {
            std::cout << "update <model> <experiences> <model_out> <generation> <name>" << std::endl;
            return 1;
        }
        constexpr size_t arg_model = 2;
        constexpr size_t arg_experiences = 3;
        constexpr size_t arg_model_out = 4;
        constexpr size_t arg_generation = 5;
        constexpr size_t arg_name = 6;
        std::cout << "Update!" << std::endl;
        std::cout << "Learning rate: " << LR << std::endl;
        std::experimental::filesystem::create_directories("logs/");
        debug_log.open(std::string("logs/overmind") + argv[arg_generation] + ".log");
        Benchmark full_ud("full_update");
        Model m(LR);
        Benchmark full_ud3("full_update3");
        if(!m.load_file(argv[2]))
        {
            std::cout << "The horse has no carrot" << std::endl;
            return 1;
        }
        Benchmark full_ud2("full_update2");

        static std::vector<Model> experiences;
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
        float mean_reward = 0;
        std::cout << "Starting updates" << std::endl;
        Model agg_experiences(LR);
        {
            Benchmark exp_agg("Experience aggregation");
            for(auto &e : experiences)
            {
                float e_mean_reward = calculate_rewards(e);
                mean_reward += e_mean_reward;
                agg_experiences.append_experience(e);
            }
            // distill(agg_experiences);
            mean_reward /= static_cast<float>(experiences.size());
        }
		for(int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
			Benchmark bepoch{"epoch"};
			m.optimizer.zero_grad();
            m.value_optimizer.zero_grad();
            // for(auto &e : experiences)
            // {
            update_model_softmax(m, agg_experiences, sm, 0.0, false);
            // }
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
        analyze_step(m, experiences[0]);
#ifndef NO_HAMPUS
        {
            Benchmark hampe_dbg("hampe_dbg");
            rds_data rds;

            rds["rewards"] = &experiences[0].rewards;
            rds["actions"] = &experiences[0].actions;

            auto np = m.net->named_parameters();
            auto oldp = experiences[0].net->named_parameters();

            for(auto &ref : np.pairs())
            {
                std::cout << ref.first << std::endl;
                std::cout << "mean: " << ref.second.mean().item<float>() << " std: " << ref.second.std().item<float>() << std::endl;
                auto cp = ref.second.to(torch::kCPU);
                auto dcp = (ref.second - oldp[ref.first]).to(torch::kCPU);
                //
                // Pointless dumb memcpy...
                //
                int64_t nel = std::min(cp.numel(), static_cast<int64_t>(1000));
                std::vector<float> p(cp.data_ptr<float>(), cp.data_ptr<float>() + nel);
                std::vector<float> dp(dcp.data_ptr<float>(), dcp.data_ptr<float>() + nel);

                rds["parameters"][ref.first]["values"] = &p;
                rds["dparameters"][ref.first]["values"] = &dp;
                rds["parameter_stats"][ref.first]["mean"] = ref.second.mean().item<float>();
                rds["parameter_stats"][ref.first]["stddev"] = ref.second.std().item<float>();
            }

            rds["mean_reward"] = mean_reward;

            std::vector<float> values;
            values.resize(experiences[0].get_frames());
            for(int frame = 0; frame < experiences[0].get_frames(); frame+=BATCH_SIZE)
            {
                size_t actual_bs = std::min(BATCH_SIZE, experiences[0].get_frames() - frame);
                if(actual_bs == 1) {
                    break;
                }
                auto v = m.value_net->forward(experiences[0].get_batch(frame, frame + actual_bs)).to(torch::kCPU);
                std::copy(v.data_ptr<float>(), v.data_ptr<float>() + actual_bs, values.begin() + frame);
            }
            rds["values"] = &values;

            std::stringstream tmp_file;
            std::stringstream metric_file;
            std::experimental::filesystem::path p = std::experimental::filesystem::current_path();
            std::experimental::filesystem::create_directories(p/"metrics/");
            std::experimental::filesystem::create_directories(p/"tmp_metrics/");
            tmp_file << std::string("tmp_metrics/") << argv[arg_name] << "_" << argv[arg_generation] << ".rds";
            metric_file << std::string("metrics/") << argv[arg_name] << "_" << argv[arg_generation] << ".rds";
            {
                std::ofstream out(tmp_file.str());
                rds.save(out);

                std::stringstream ss;
                ss << "gzip -f " << tmp_file;
                system(ss.str().c_str());
            }
            std::experimental::filesystem::rename(p/tmp_file.str(), p/metric_file.str());
        }
#endif
        {
            Benchmark save("savefile");
            m.save_file(argv[4]);
        }
    }
    else
    {
        std::cout << "unknown command" << std::endl;
        return 1;
    }
    return 0;
}
