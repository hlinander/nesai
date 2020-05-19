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
    int batch_size = 4;
	if(bs)
	{
		batch_size = atoi(bs);
	}
    std::cout << "BATCH_SIZE " << batch_size << std::endl;
	return batch_size;
}

static int get_ppo_epochs()
{
	const char *bs = getenv("PPO_EPOCHS");
    int ppo = 1;
	if(bs)
	{
		ppo = atoi(bs);
	}
    std::cout << "PPOEPOCHS " << ppo << std::endl;
	return ppo;
}

static float get_learning_rate()
{
    const float default_lr = 0.00001;
	const char *lr = getenv("LR");
	if(lr)
	{
        try {
		    float parsed_lr = std::stof(lr);
            std::cout << "LR from env: " << parsed_lr << std::endl;
            return parsed_lr;
        }
        catch(const std::invalid_argument& ia)
        {
            std::cout << "INVALID LEARNING RATE" << std::endl;
            return default_lr;
        }
	}
    std::cout << "LR: " << default_lr << std::endl;
	return default_lr;
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

void save_state(StateType &s, std::string filename)
{
    std::ofstream f(filename);
    f << "P2" << std::endl;
    f << "32 " << int(STATE_SIZE / 32) << std::endl;
    f << "255" << std::endl;
    for(int i = 0; i < STATE_SIZE; ++i)
    {
        uint8_t value = static_cast<uint8_t>(255.0 * (s[i] + 0.5));
        f << std::to_string(value) + std::string(" ");
        if(i % 32 == 0)
            f << std::endl;
    }
}

void test_update_model();
void test_decoder();

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

std::array<float, 4> update_model_softmax(Model &m, Model &experience, size_t bs, size_t batch_size, size_t unroll_size) {
    m.net->train();
    m.optimizer.zero_grad();
    int it = 0;
    float total_loss = 0.0f;
    float value_loss = 0.0f;
    float reward_loss = 0.0f;
    float policy_loss = 0.0f;
    std::vector<std::vector<std::vector<size_t>>> batches;
    std::vector<StateType> batch_data;
    std::vector<ActionType> batch_actions; 
    std::vector<float> batch_rewards;
    std::vector<float> batch_immidiate_rewards;
    batch_data.resize(bs);
    batch_actions.resize(bs);
    batch_rewards.resize(bs);
    batch_immidiate_rewards.resize(bs);
    size_t n_starts = experience.get_frames() - unroll_size;
    std::cout << "e frames " << experience.get_frames() << std::endl << std::flush;
    size_t n_batches = std::max(static_cast<unsigned int>(n_starts / (bs * batch_size)), 1U);
    std::cout << "n batches " << n_batches << std::endl << std::flush;
    batches.resize(n_batches);
    for(auto &batch: batches)
    {
        batch.resize(batch_size);
        for(auto& starts: batch) {
            starts.resize(bs);
            for(auto& start: starts)
            {
                float frac = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                start = static_cast<size_t>(frac * (experience.get_frames() - unroll_size));
            }
        }
    }
    std::cout << "[Overmind] Batches: " << batches.size() << " batch size: " << batch_size << std::endl;
	for (auto& batch: batches)
    {
        std::cout << "." << std::flush;
        torch::Tensor loss = torch::tensor({0.0f}).to(experience.net->device);
        for(auto& starts: batch)
        {
            size_t n = 0;
            for(auto frame: starts) {
                std::copy(experience.states[frame].begin(), experience.states[frame].end(), batch_data[n].begin());
                ++n;
            }
            //Output next = m.initial_forward(experience.states[frame]);
            Output next = m.initial_forward_batched(batch_data);

            for(int sim = 0; sim < unroll_size; ++sim) {
                size_t n = 0;
                for(auto frame: starts) {
                    std::copy(experience.actions[frame + sim].begin(), experience.actions[frame + sim].end(), batch_actions[n].begin());
                    batch_rewards[n] = experience.rewards[frame + sim];
                    batch_immidiate_rewards[n] = experience.immidiate_rewards[frame + sim];
                    ++n;
                }
                auto device_rewards = device_tensor_batched(static_cast<void*>(batch_rewards.data()), batch_rewards.size(), experience.net->device);
                auto device_immidiate_rewards = device_tensor_batched(static_cast<void*>(batch_immidiate_rewards.data()), batch_rewards.size(), experience.net->device);
                device_rewards = device_rewards.reshape({batch_rewards.size(), 1});
                device_immidiate_rewards = device_immidiate_rewards.reshape({batch_rewards.size(), 1});
                if(sim > 0) {
                    next = m.net->recurrent_forward_batched(next.hidden, batch_actions);
                }
                print_size("next.reward", next.reward);
                print_size("next.value", next.value);
                print_size("next.policy", next.policy);
                auto action = device_tensor_batched(static_cast<void*>(batch_actions.data()), batch_actions.size() * ACTION_SIZE, experience.net->device, torch::kUInt8, torch::kLong);
                action = action.reshape({batch_actions.size(), ACTION_SIZE});
                auto action_tensor = torch::argmax(action, 1, true).reshape({batch_actions.size()});
                print_size("Action tensor ", action_tensor);
                auto r_loss = ((next.reward - device_immidiate_rewards) * (next.reward - device_immidiate_rewards)).mean();
                loss += r_loss;
                auto v_loss = ((next.value - device_rewards) * (next.value - device_rewards)).mean();
                loss += v_loss;
                auto sm = torch::log_softmax(next.policy, 1);
                print_size("softmax tensor", sm);
                auto p_loss = torch::nll_loss(sm, action_tensor);
                loss += p_loss;
                // std::cout << "After loss" << std::endl;
                total_loss += loss.item<float>();
                value_loss += v_loss.item<float>();
                policy_loss += p_loss.item<float>();
                reward_loss += r_loss.item<float>();
            }
            ++it;
        }
        loss.backward();
	}

    m.optimizer.step();
    float t = static_cast<float>(it);
    return std::array<float, 4>{total_loss / t, value_loss / t, reward_loss / t, policy_loss / t};
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
    std::experimental::filesystem::create_directories("mcts/");
    std::experimental::filesystem::create_directories("states/");
    std::experimental::filesystem::create_directories("plots/");

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
        auto np = m.net->named_parameters();

        for(auto &ref : np.pairs())
        {
            std::cout << ref.first << std::endl;
            std::cout << "sizes: [";
            for(auto &it: ref.second.sizes())
                std::cout << it << ", ";
            std::cout << std::endl;
            std::cout << "mean: " << ref.second.mean().item<float>() << " std: " << ref.second.std().item<float>() << std::endl;
        }
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
        std::unique_ptr<Model> exp = std::make_unique<Model>(LR);
        Benchmark full_ud3("full_update3");
        if(!m.load_file(argv[2]))
        {
            std::cout << "The horse has no carrot" << std::endl;
            return 1;
        }
        Benchmark full_ud2("full_update2");

        // static std::vector<Model> experiences;
        std::vector<std::string> experience_files;
        std::ifstream in{argv[3]};
        std::string str;
        {
            Benchmark b{"exp_load"};
            std::cout << "Reading lines..." << std::endl;
            while(std::getline(in, str))
            {
                experience_files.push_back(str);
                std::cout << str << std::endl;
                // try {
                //     experiences.emplace_back(Model(LR));
                //     if(!experiences.back().load_file(str))
                //     {
                //         std::cout << "Missing exp: " << str << std::endl;
                //         return 3;
                //     }
                //     std::cout << "Loaded experience with " << experiences.back().get_frames() << " frames" << std::endl;
                // } catch(const std::exception &e) {
                //     std::cout << "FIXTHISFFS:" << e.what() << std::endl;
                // }
            }
        }

        stat_map sm;
        float mean_reward = 0;
        // std::cout << "Starting updates" << std::endl;
        // {
        //     Benchmark exp_agg("Experience aggregation");
        //     for(auto &e : experiences)
        //     {
        //         float e_mean_reward = calculate_rewards(e, DISCOUNT);
        //         mean_reward += e_mean_reward;
        //         // agg_experiences.append_experience(e);
        //     }
        //     // distill(agg_experiences);
        //     mean_reward /= static_cast<float>(experiences.size());
        // }
        std::array<float, 4> losses;
        for(auto &ef : experience_files)
        {
            //Model e(LR);
            exp.reset();
            exp = std::make_unique<Model>(LR);
            exp->load_file(ef);
            mean_reward += calculate_rewards(*exp, DISCOUNT);
            losses = update_model_softmax(m, *exp, 5, 512, 5);
        }
        mean_reward /= static_cast<float>(experience_files.size());
        // }
        // m.optimizer.step();
        // m.value_optimizer.step();
        // if(m.net->isnan()) {
        //     std::cout << "Doomed tensors!" << std::endl;
        //     for(auto &e : experiences)
        //     {
        //         std::cout << "experience" << std::endl;
        //         for(int i = 0; i < e.get_frames(); ++i)
        //         {
        //             if(fabs(e.immidiate_rewards[i]) > 0.0001)
        //                 doom_log << e.immidiate_rewards[i] << std::endl;
        //         }
        //     }

        // }
        // analyze_step(m, experiences[0]);
#ifndef NO_HAMPUS
        {
            int i = 0;
            for(auto &s: exp->states)
            {
                save_state(s, std::string("states/") + std::to_string(i) + std::string(".pgm"));
                ++i;
            }
            Benchmark hampe_dbg("hampe_dbg");
            rds_data rds;

            rds["rewards"] = &exp->rewards;
            // for(auto &it: exp.rewards)
            //     std::cout << it << ", " << std::endl;
            rds["normalized_rewards"] = &exp->normalized_rewards;
            rds["immidiate_rewards"] = &exp->immidiate_rewards;
            rds["actions"] = &exp->actions;
            rds["predicted_values"] = &exp->predicted_values;
            // for(auto &it: exp->predicted_values)
            //     std::cout << it << ", " << std::endl;
            rds["predicted_rewards"] = &exp->predicted_rewards;
            std::vector<float> state_data(exp->states[0].begin(), exp->states[0].end());
            rds["state"] = &state_data;

            auto np = m.net->named_parameters();
            auto oldp = exp->net->named_parameters();
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
            rds["loss"] = losses[0];
            rds["reward_loss"] = losses[1];
            rds["value_loss"] = losses[2];
            rds["policy_loss"] = losses[3];

            {
                std::stringstream json_metric_file;
                json_metric_file << std::string("metrics/") << argv[arg_name] << "_" << std::setfill('0') << std::setw(5) << argv[arg_generation] << ".json";
                std::ofstream out(json_metric_file.str());
                out << std::setw(4) << json << std::endl;
            }

            // std::vector<float> predicted_values;
            // predicted_values.resize(experiences[0].get_frames());
            // demi
            // for(int frame = 0; frame < experiences[0].get_frames(); frame+=BATCH_SIZE)
            // {
            //     size_t actual_bs = std::min(BATCH_SIZE, experiences[0].get_frames() - frame);
            //     if(actual_bs == 1) {
            //         break;
            //     }
            //     auto v = m.value_net->forward(experiences[0].get_batch(frame, frame + actual_bs)).to(torch::kCPU);
            //     std::copy(v.data_ptr<float>(), v.data_ptr<float>() + actual_bs, predicted_values.begin() + frame);
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
    else if(0 == strcmp(argv[1], "testdecoder"))
    {
        // int result = Catch::Session().run( argc - 1, argv + 1 );
        test_decoder();
    }
    else
    {
        std::cout << "unknown command" << std::endl;
        return 1;
    }
    return 0;
}

void test_decoder() {
    constexpr size_t size = STATE_SIZE;
    std::shared_ptr<Encoder<N_HIDDEN>> e{std::make_shared<Encoder<N_HIDDEN>>()};
    std::shared_ptr<Decoder<N_HIDDEN, 1>> d{std::make_shared<Decoder<N_HIDDEN, 1>>()};
    std::shared_ptr<Net> net{std::make_shared<Net>()};
    Model m(LR);
	torch::optim::Adam eoptimizer(e->parameters(), torch::optim::AdamOptions(LR));
	torch::optim::Adam doptimizer(d->parameters(), torch::optim::AdamOptions(LR));
	torch::optim::Adam optimizer_net(net->parameters(), torch::optim::AdamOptions(LR));

    std::array<float, size> in;
    auto set = [](std::array<float, size> &s, float v) {
        std::fill(s.begin(), s.end(), v);
    };
    auto target = [](int in) -> float {
        // return -1.0 + static_cast<float>(2 * in) / 500.0;
        return static_cast<float>(in * in) / (1000.0 * 1000.0);
    };
    for(int i = 0; i < 100; ++i)
    {
        std::stringstream s;
        s << std::setw(4) << std::setfill('0') << i;
        std::ofstream plot_file(std::string("plots/") + s.str() + std::string(".dat"));
        plot_file << "in" << " " << "target" << " " << "value" << std::endl;
        for(int j = 0; j < 1000; j+= 10)
        {
            set(in, static_cast<float>(j) / 1000.0);
            auto output = m.initial_forward(in);
            auto v = output.value.item<float>();
            plot_file << std::to_string(j) << " " << std::to_string(target(j)) << " " << std::to_string(v) << std::endl;
        }
        plot_file.close();
        eoptimizer.zero_grad();
        doptimizer.zero_grad();
        optimizer_net.zero_grad();
        m.optimizer.zero_grad();
        float total_loss = 0.0f;
        float total_loss_net = 0.0f;
        float total_loss_m = 0.0f;
        for(int j = 0; j < 1000; j++)
        {
            set(in, static_cast<float>(j) / 1000.0);
            auto t = device_tensor(in, torch::kCPU).reshape({1, size});
            auto out = d->forward(e->forward(t));
            auto output_net = net->initial_forward(t);
            auto output_m = m.initial_forward(in);
            auto out_net = output_net.value;
            auto out_m = output_m.value;
            auto delta = (out[0] - target(j));
            auto delta_net = (out_net[0] - target(j));
            auto delta_m = (out_m[0] - target(j));
            auto loss = delta * delta;
            auto loss_net = delta_net * delta_net;
            auto loss_m = delta_m * delta_m;
            loss.backward();
            loss_net.backward();
            loss_m.backward();
            total_loss += loss.item<float>();
            total_loss_net += loss_net.item<float>();
            total_loss_m += loss_m.item<float>();
        }
        eoptimizer.step();
        doptimizer.step();
        optimizer_net.step();
        m.optimizer.step();
        std::cout << "Encoder + Decoder: " << total_loss << std::endl;
        std::cout << "Net: " << total_loss_net << std::endl;
        std::cout << "Model: " << total_loss_m << std::endl;
    }
}

void test_update_model() {
    Model m1(LR);
    Model mold(LR);
    Model exp(LR);
    m1.save_file("/tmp/m");
    exp.load_file("/tmp/m");
    StateType s1, s2;
    ActionType a1, a2;
    a1.fill(0u);
    a1[1] = 1u;
    a2.fill(0u);
    a2[0] = 1u;
    auto set = [](StateType &s, float v) {
        std::fill(s.begin(), s.end(), v);
    };

    auto target = [](int in) {
        return -1.0 + static_cast<float>(2 * in) / 500.0;
    };
    // for(size_t i = 0; i < STATE_SIZE; ++i)
    // {
    //     s1[i] = .1f;
    //     s2[i] = -.1f;
    // }
    for(int i = 0; i < 250; ++i)
    {
        set(s1, target(i));
        exp.record_action(s1, a1, 0.01f, 1.0f, 1.0f);
    }
    for(int i = 250; i < 500; ++i)
    {
        set(s1, target(i));
        exp.record_action(s1, a1, 0.02f, 1.0f, 1.0f);
    }
    calculate_rewards(exp, DISCOUNT);
    for(int i = 0; i < exp.rewards.size(); i+=10)
        std::cout << i << ": " << exp.rewards[i] << std::endl;
    std::ofstream loss_file(std::string("plots/losses"));
    loss_file << "total value reward policy" << std::endl;
    for(int i = 0; i < 1000; ++i)
    {
        // m1.optimizer.zero_grad();
        std::stringstream s;
        s << std::setw(4) << std::setfill('0') << i;
        std::ofstream plot_file(std::string("plots/") + s.str() + std::string(".dat"));
        plot_file << "in" << " " << "target" << " " << "value" << std::endl;
        for(int j = 0; j < 500; j+= 10)
        {
            set(s1, target(j));
            // std::cout << "Test initial forward" << std::endl;
            auto output = m1.initial_forward(s1);
            // std::cout << "After initial forward" << std::endl;
            auto v = output.value.item<float>();
            plot_file << std::to_string(target(j)) << " " << std::to_string(exp.rewards[j]) << " " << std::to_string(v) << std::endl;
        }
        plot_file.close();
        m1.save_file("/tmp/m");
        mold.load_file("/tmp/m");
        auto mold_np = mold.net->named_parameters();
        auto losses = update_model_softmax(m1, exp, 10, 100, 1);
        loss_file << losses[0] << " " << losses[1] << " " << losses[2] << " " << losses[3] << std::endl << std::flush;
        std::cout << losses[0] << " " << losses[1] << " " << losses[2] << " " << losses[3] << std::endl << std::flush;
        for(auto &it: m1.net->named_parameters().pairs())
        {
            // std::cout << "diff " << it.first << ": " << torch::abs((it.second - mold_np[it.first])).mean().item<float>();
            // std::cout << std::endl;
        }
        // m1.optimizer.step();
        // std::cout << ".";
        // if(i % 1 == 0) {
        //     set(s1, -1.0f);
        //     auto output1 = m1.initial_forward(s1);
        //     set(s1, 1.0f);
        //     auto output2 = m1.initial_forward(s1);
        //     // auto pa = torch::softmax(output1.policy, 1).to(torch::kCPU);
        //     auto r = output1.reward.item<float>();
        //     auto v1 = output1.value.item<float>();
        //     auto v2 = output2.value.item<float>();
        //     // auto s1a = m1.get_value(s1);
        //     // auto s2a = m1.get_value(s2);
        //     // auto paa = pa.accessor<float, 2>();
        //     // std::cout << "[";
        //     // for(int i = 0; i < ACTION_SIZE; ++i)
        //     // {
        //     //     std::cout << paa[0][i] << ", ";
        //     // }
        //     // std::cout << std::endl << std::flush;
        //     std::cout << "] " 
        //         << std::endl 
        //         // << " r: " << r
        //         << " v1: " << v1 << " true: " << exp.rewards[0]
        //         << " v2: " << v2 << " true: " << exp.rewards[499]
        //         << std::endl << std::flush;
        // }
    }
}
