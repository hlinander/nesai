#define CATCH_CONFIG_RUNNER
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
#include "catch.hpp"

#define DEBUG(...)

static int get_batch_size()
{
	const char *bs = getenv("BATCH_SIZE");
	if(bs)
	{
		return atoi(bs);
	}
	return 5;
}

static int get_ppo_epochs()
{
	const char *bs = getenv("PPO_EPOCHS");
	if(bs)
	{
		return atoi(bs);
	}
	return 1;
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

static float get_discount()
{
    const float default_discount = 0.8;
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

const float DISCOUNT = get_discount();
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

void test_update_model();

// void analyze_step(Model &after, Model &experience) {
// 	// Reward reward = calculate_rewards(experience);
//     after.net->eval();
//     experience.net->eval();
//     for(size_t frame = 0; frame < (size_t)experience.get_frames(); ++frame)
//     {
//         auto prob_before = torch::softmax(experience.forward(experience.states[frame]), 1).to(torch::kCPU);
//         auto prob_after = torch::softmax(after.forward(experience.states[frame]), 1).to(torch::kCPU);
//         auto pb = prob_before.accessor<float, 2>();
//         auto pa = prob_after.accessor<float, 2>();

//         debug_log << "Reward: " << experience.rewards[frame] << " ";
//         for(int i = 0; i < prob_before.sizes()[1]; ++i)
//         {
//             if(experience.actions[frame][i] == 1)
//             {
//                 debug_log << " **[" << pb[0][i] << " -> " << pa[0][i] << "]**, ";
//             }
//             else
//             {
//                 // debug_log << pb[0][i] << " -> " << pa[0][i] << ", ";
//             }
//         }
//         debug_log << std::endl;
//         // debug_log << "Reward: " << experience.rewards[frame] << std::endl;
//         // debug_log << "Before: " << prob_before << std::endl;
//         // debug_log << "After: " << prob_after << std::endl;
//     }
//     debug_log << std::endl << std::flush;
// }

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

void update_model_softmax(Model &m, Model &experience) {
    m.net->train();
	for (int frame = 0; frame < experience.get_frames(); frame+=BATCH_SIZE) {
        // std::cout << "#" << std::flush;
		size_t actual_bs = std::min(BATCH_SIZE, experience.get_frames() - frame);
        if(actual_bs == 1) {
            break;
        }
        torch::Tensor loss = torch::tensor({0.0f});
        // Output next = m.net->initial_forward(experience.get_state(frame));
        Output next = m.initial_forward(experience.states[frame]);

        for(int sim = 0; sim < BATCH_SIZE; ++sim) {
            if(sim > 0) {
                next = m.net->recurrent_forward(next.hidden, experience.actions[frame + sim]);
            }
            // for(auto &it: experience.actions[frame + sim])
            //     std::cout << int(it) << ", ";
            auto action = device_tensor(experience.actions[frame + sim], experience.net->device, torch::kUInt8, torch::kLong);
            // std::cout << action << std::endl;
            auto action_tensor = torch::argmax(action, c10::nullopt, true).reshape({1});
            // std::cout << action_tensor[0].item<float>() << " : ";
            // std::cout << "I: " << experience.immidiate_rewards[frame + sim] << std::endl;
            // std::cout << "R: " << experience.rewards[frame + sim] << std::endl;
            loss += (next.reward[0] - experience.immidiate_rewards[frame + sim]) * (next.reward[0] - experience.immidiate_rewards[frame + sim]);
            loss += (next.value[0] - experience.rewards[frame + sim]) * (next.value[0] - experience.rewards[frame + sim]);
            auto sm = torch::log_softmax(next.policy, 1);
            // std::cout << sm << std::endl;
            loss += torch::nll_loss(sm, action_tensor);
        }
        loss.backward();
	}

	DEBUG("Loss backwards\n");
	DEBUG("Returning...\n");
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
        replay(m, DISCOUNT);
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
        std::stringstream gen_str;
        gen_str << std::setfill('0') << std::setw(5) << std::stoi(argv[arg_generation]);
        debug_log.open(std::string("logs/overmind_") + argv[arg_name] + "_" + gen_str.str() + ".log");
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
                float e_mean_reward = calculate_rewards(e, DISCOUNT);
                mean_reward += e_mean_reward;
                agg_experiences.append_experience(e);
            }
            // distill(agg_experiences);
            mean_reward /= static_cast<float>(experiences.size());
        }
		for(int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
			Benchmark bepoch{"epoch"};
			m.optimizer.zero_grad();
            // m.value_optimizer.zero_grad();
            // for(auto &e : experiences)
            // {
            update_model_softmax(m, agg_experiences);
            // }
			m.optimizer.step();
			// m.value_optimizer.step();
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
        // analyze_step(m, experiences[0]);
#ifndef NO_HAMPUS
        {
            Benchmark hampe_dbg("hampe_dbg");
            rds_data rds;

            rds["rewards"] = &experiences[0].rewards;
            rds["normalized_rewards"] = &experiences[0].normalized_rewards;
            rds["advantage"] = &experiences[0].adv;
            rds["actions"] = &experiences[0].actions;
            rds["values"] = &experiences[0].values;

            auto np = m.net->named_parameters();
            auto oldp = experiences[0].net->named_parameters();
            std::list<std::vector<float>> p_list;
            std::list<std::vector<float>> dp_list;

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
                p_list.emplace_back(std::vector<float>(cp.data_ptr<float>(), cp.data_ptr<float>() + nel));
                dp_list.emplace_back(std::vector<float>(dcp.data_ptr<float>(), dcp.data_ptr<float>() + nel));

                rds["parameters"][ref.first]["values"] = &p_list.back();
                rds["dparameters"][ref.first]["values"] = &dp_list.back();
                rds["parameter_stats"][ref.first]["mean"] = ref.second.mean().item<float>();
                rds["parameter_stats"][ref.first]["stddev"] = ref.second.std().item<float>();
            }

            rds["mean_reward"] = mean_reward;
            json["mean_reward"] = mean_reward;

            {
                std::stringstream json_metric_file;
                json_metric_file << std::string("metrics/") << argv[arg_name] << "_" << std::setfill('0') << std::setw(5) << argv[arg_generation] << ".json";
                std::ofstream out(json_metric_file.str());
                out << std::setw(4) << json << std::endl;
            }

            // std::vector<float> values;
            // values.resize(experiences[0].get_frames());
            // for(int frame = 0; frame < experiences[0].get_frames(); frame+=BATCH_SIZE)
            // {
            //     size_t actual_bs = std::min(BATCH_SIZE, experiences[0].get_frames() - frame);
            //     if(actual_bs == 1) {
            //         break;
            //     }
            //     auto v = m.value_net->forward(experiences[0].get_batch(frame, frame + actual_bs)).to(torch::kCPU);
            //     std::copy(v.data_ptr<float>(), v.data_ptr<float>() + actual_bs, values.begin() + frame);
            // }

            std::stringstream tmp_file;
            std::stringstream tmp_file_gz;
            std::stringstream metric_file;
            std::experimental::filesystem::path p = std::experimental::filesystem::current_path();
            std::experimental::filesystem::create_directories(p/"metrics/");
            std::experimental::filesystem::create_directories(p/"tmp_metrics/");
            tmp_file << std::string("tmp_metrics/") << argv[arg_name] << "_" << std::setfill('0') << std::setw(5) << argv[arg_generation] << ".rds";
            tmp_file_gz << std::string("tmp_metrics/") << argv[arg_name] << "_" << std::setfill('0') << std::setw(5) << argv[arg_generation] << ".rds.gz";
            metric_file << std::string("metrics/") << argv[arg_name] << "_" << std::setfill('0') << std::setw(5) << argv[arg_generation] << ".rds";
            {
                std::ofstream out(tmp_file.str());
                rds.save(out);
            }

            std::stringstream ss;
            ss << "gzip -f " << tmp_file.str();
            system(ss.str().c_str());
            std::experimental::filesystem::rename(p/tmp_file_gz.str(), p/metric_file.str());
        }
#endif
        {
            Benchmark save("savefile");
            m.save_file(argv[4]);
        }
    }
    else if(0 == strcmp(argv[1], "test"))
    {
        int result = Catch::Session().run( argc - 1, argv + 1 );
        test_update_model();
    }
    else
    {
        std::cout << "unknown command" << std::endl;
        return 1;
    }
    return 0;
}

void test_update_model() {
    Model m1(LR);
    Model exp(LR);
    m1.save_file("/tmp/m");
    exp.load_file("/tmp/m");
    StateType s1, s2;
    ActionType a1, a2;
    a1.fill(0u);
    a1[1] = 1u;
    a2.fill(0u);
    a2[0] = 1u;
    for(size_t i = 0; i < STATE_SIZE; ++i)
    {
        s1[i] = .1f;
        s2[i] = -.1f;
    }
    for(int i = 0; i < 500; ++i)
    {
        exp.record_action(s1, a1, 100.0f, 1.0f);
        m1.record_action(s1, a1, 100.0f, 1.0f);
    }
    for(int i = 0; i < 500; ++i)
    {
        exp.record_action(s2, a2, -100.0f, 1.0f);
        m1.record_action(s2, a2, -100.0f, 1.0f);
    }
    calculate_rewards(exp, DISCOUNT);
    calculate_rewards(m1, DISCOUNT);
    for(int i = 0; i < 200; ++i)
    {
        m1.optimizer.zero_grad();
        update_model_softmax(m1, exp);
        m1.optimizer.step();
        // std::cout << ".";
        m1.save_file("/tmp/m");
        exp.load_file("/tmp/m");
        if(i % 1 == 0) {
            auto output = m1.initial_forward(s1);
            auto pa = torch::softmax(output.policy, 1).to(torch::kCPU);
            auto r = output.reward.item<float>();
            auto v = output.value.item<float>();
            // auto s1a = m1.get_value(s1);
            // auto s2a = m1.get_value(s2);
            auto paa = pa.accessor<float, 2>();
            std::cout << "[";
            for(int i = 0; i < ACTION_SIZE; ++i)
            {
                std::cout << paa[0][i] << ", ";
            }
            // std::cout << std::endl << std::flush;
            std::cout << "] " 
                << std::endl 
                << " r: " << r
                << " v: " << v 
                << std::endl << std::flush;
        }
    }
}