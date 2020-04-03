#pragma once

#include <torch/torch.h>
#include <vector>
#include <array>
#include <map>
#include <iostream>
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/list.hpp>
#include "bm.h"
#include "reward.h"

const int N_HIDDEN = 64U;
const int RAM_SIZE = 0x800;
const int SCREEN_W = 32;
const int SCREEN_H = 30;
const int SCREEN_PIXELS = SCREEN_W*SCREEN_H;
const int STATE_SIZE = RAM_SIZE + (SCREEN_PIXELS * 3);
const float ACTION_THRESHOLD = 0.5f;
const float EPS_EXPLORE = 0.01f;
const float GAMMA = get_discount();

enum class Action
{
	UP = 0,
	DOWN,
	LEFT,
	RIGHT,
	UP_A,
	UP_B,
	DOWN_A,
	DOWN_B,
	LEFT_A,
	LEFT_B,
	RIGHT_A,
	RIGHT_B,
	A,
	B,
	START,
	SELECT,
	MAX
};

const std::vector<std::string> action_names = {
	"UP",
	"DOWN",
	"LEFT",
	"RIGHT",
	"UP_A",
	"UP_B",
	"DOWN_A",
	"DOWN_B",
	"LEFT_A",
	"LEFT_B",
	"RIGHT_A",
	"RIGHT_B",
	"A",
	"B",
	"START",
	"SELECT"};

constexpr size_t ACTION_SIZE = static_cast<size_t>(Action::MAX);

typedef std::array<float, STATE_SIZE> StateType;
typedef std::array<uint8_t, ACTION_SIZE> ActionType;

static torch::Device get_device()
{
	std::cout << "!!!!!!!!!!!!FORCECPU!!!!!!!!!!!" << std::endl;
	return torch::kCPU;
	if (getenv("NO_CUDA"))
	{
		return torch::kCPU;
	}
	if (torch::cuda::is_available())
	{
		std::cout << "THERE IS A GPU" << std::endl;
		return torch::kCUDA;
	}
	else
	{
		return torch::kCPU;
	}
}

template <typename T>
torch::Tensor device_tensor(T &container, torch::Device device, c10::ScalarType in_type = torch::kFloat32, c10::ScalarType out_type = torch::kFloat32)
{
	auto ret = torch::from_blob(static_cast<void *>(container.data()), {container.size()}, in_type).to(out_type);
	return ret.to(device);
}

template <int N_OUTPUT>
struct Encoder : torch::nn::Module
{
	Encoder() : device(get_device())
	{
		Benchmark nt("Net constrct");
		fc1 = register_module("fc1", torch::nn::Linear(STATE_SIZE, N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_HIDDEN, N_OUTPUT));
		// fc3 = register_module("fc3", torch::nn::Linear(N_HIDDEN, N_OUTPUT));
		torch::nn::init::xavier_normal_(fc1->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
		// torch::nn::init::xavier_normal_(fc3->weight);
		// torch::nn::init::zeros_(fc3->bias);
		torch::nn::init::zeros_(fc2->bias);
		torch::nn::init::zeros_(fc1->bias);
		Benchmark togpu("NET to GPU");
		to(device);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = (torch::leaky_relu(fc1->forward(x)));
		x = (torch::leaky_relu(fc2->forward(x)));
		// x = fc3->forward(x);
		return x;
	}

	bool isnan() const
	{
		for (auto &t : parameters())
		{
			if (at::any(at::isnan(t)).item<bool>())
			{
				return true;
			}
		}
		return false;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	torch::Device device;
};

template <int N_INPUT, int N_OUTPUT>
struct Decoder : torch::nn::Module
{
	Decoder() : device(get_device())
	{
		Benchmark nt("Net constrct");
		// fc1 = register_module("fc1", torch::nn::Linear(STATE_SIZE, N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_INPUT, N_HIDDEN));
		fc3 = register_module("fc3", torch::nn::Linear(N_HIDDEN, N_OUTPUT));
		// torch::nn::init::xavier_normal_(fc1->weight);
		// torch::nn::init::xavier_normal_(fc2->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
		torch::nn::init::zeros_(fc2->bias);
		torch::nn::init::xavier_normal_(fc3->weight);
		torch::nn::init::zeros_(fc3->bias);
		// torch::nn::init::zeros_(fc2->bias);
		// torch::nn::init::zeros_(fc1->bias);
		Benchmark togpu("NET to GPU");
		to(device);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		// x = (torch::leaky_relu(fc1->forward(x)));
		// x = (torch::leaky_relu(fc2->forward(x)));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}

	bool isnan() const
	{
		for (auto &t : parameters())
		{
			if (at::any(at::isnan(t)).item<bool>())
			{
				return true;
			}
		}
		return false;
	}

	torch::nn::Linear /* fc1{nullptr}, */ fc2{nullptr}, fc3{nullptr};
	torch::Device device;
};

struct Output
{
	torch::Tensor policy;
	torch::Tensor value;
	torch::Tensor reward;
	torch::Tensor hidden;
};

struct Net : torch::nn::Module
{
	Net() : device(get_device())
	{
		encoder = register_module("Encoder", std::make_shared<Encoder<N_HIDDEN>>());
		dynamics = register_module("Dynamics", std::make_shared<Decoder<N_HIDDEN + ACTION_SIZE, N_HIDDEN>>());
		policy = register_module("Policy", std::make_shared<Decoder<N_HIDDEN, ACTION_SIZE>>());
		value = register_module("Value", std::make_shared<Decoder<N_HIDDEN, 1>>());
		reward = register_module("Reward", std::make_shared<Decoder<N_HIDDEN, 1>>());
		to(device);
	}

	bool isnan()
	{
		return encoder->isnan() | dynamics->isnan() | policy->isnan() | value->isnan() | reward->isnan();
	}

	Output initial_forward(torch::Tensor state)
	{
		auto features = encoder->forward(state);
		return Output({policy->forward(features),
					   value->forward(features),
					   reward->forward(features),
					   features});
	}

	Output recurrent_forward(torch::Tensor features, ActionType action)
	{
		// auto features = net->encoder->forward(dts);
		auto action_tensor = device_tensor(action, device, torch::kUInt8, torch::kFloat32).reshape({1, action.size()});
		auto features_and_actions = torch::cat({features, action_tensor}, -1);
		return Output({policy->forward(features),
					   value->forward(features),
					   reward->forward(features),
					   dynamics->forward(features_and_actions)});
	}

	std::shared_ptr<Encoder<N_HIDDEN>> encoder{nullptr};
	std::shared_ptr<Decoder<N_HIDDEN + ACTION_SIZE, N_HIDDEN>> dynamics{nullptr};
	std::shared_ptr<Decoder<N_HIDDEN, 1>> value{nullptr};
	std::shared_ptr<Decoder<N_HIDDEN, 1>> reward{nullptr};
	std::shared_ptr<Decoder<N_HIDDEN, ACTION_SIZE>> policy{nullptr};
	torch::Device device;
};

struct Model;

struct MCTSNode {
	struct params {
		params(float gamma) : 
			gamma(gamma),
			qmax(-INFINITY),
			qmin(INFINITY) {}

		std::vector<MCTSNode *> path;
		float gamma;
		float qmin;
		float qmax;
	};
	torch::Tensor hidden;
	size_t N;
	float R;
	float Q;
	float Qnorm;
	std::array<float, ACTION_SIZE> P;
	std::vector<std::unique_ptr<MCTSNode>> children;

	MCTSNode();

	void init_root(Model &m, StateType &s);
	void populate(Output o);
	void one_simulation(Model &m, params &p);
private:
	void step(Model &m, params &p);
public:
	float nsum() const;
	// children[action_idx] = 
	void update_statistics(params &p);
	size_t sample_action(float T) const;
	std::string to_dot_internal(int idx, int depth);
	void to_dot(std::string out_file);
};


struct Model
{
	// typedef Net<ACTION_SIZE> NetType;
	Model(float lr) : net{std::make_shared<Net>()},
					  optimizer(net->parameters(), torch::optim::AdamOptions(lr)),
					  n_saved_trees(0)

	//   value_optimizer(value_net->parameters(), torch::optim::AdamOptions(0.01))
	{
	}

	friend std::ostream &operator<<(std::ostream &os, const Model &m)
	{
		os << "Model took " << m.actions.size() << " actions " << std::endl;
		m.net->pretty_print(os);
		std::cout << m.net->parameters() << std::endl;
		return os;
	}

	template <class Archive>
	void save(Archive &a) const
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("values", values));
		std::stringstream ss;
		torch::save(net, ss);
		a(cereal::make_nvp("net", ss.str()));
		std::stringstream sso;
		torch::save(optimizer, sso);
		a(cereal::make_nvp("opt", sso.str()));
	}

	void save_file(const std::string &filename) const
	{
		std::stringstream ss;
		cereal::BinaryOutputArchive ar{ss};
		{
			// Benchmark s("Serialization");
			save(ar);
		}
		{
			// Benchmark f("Flush");
			std::ofstream out(filename, std::ios_base::binary);
			auto serial{ss.str()};
			out.write(serial.c_str(), serial.length());
		}
	}

	bool load_file(const std::string &filename)
	{
		std::ifstream in(filename, std::ios_base::binary);
		if (in.is_open())
		{
			cereal::BinaryInputArchive ar{in};
			load(ar);
			return true;
		}
		return false;
	}

	template <class Archive>
	void load(Archive &a)
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		a(cereal::make_nvp("values", values));
		std::string s;
		a(s);
		std::stringstream ss{s};
		torch::load(net, ss, net->device);
		std::string so;
		a(so);
		std::stringstream sso{so};
		torch::load(optimizer, sso, net->device);
	}
	/*
		encoder = register_module("Encoder", std::make_shared<Encoder<N_HIDDEN>>());
		dynamics = register_module("Dynamics", std::make_shared<Decoder<N_HIDDEN + ACTION_SIZE, N_HIDDEN>>());
		policy = register_module("Policy", std::make_shared<Decoder<N_HIDDEN, ACTION_SIZE>>());
		value = register_module("Value", std::make_shared<Decoder<N_HIDDEN, 1>>());
		reward = register_module("Reward", std::make_shared<Decoder<N_HIDDEN, 1>>());
	 */

	// torch::Tensor forward(StateType &s) {
	// 	auto ts = torch::from_blob(static_cast<void*>(s.data()), {1, STATE_SIZE}, torch::kFloat32);
	// 	auto dts = ts.to(net->device);
	// }

	// torch::Tensor forward_batch_nice(size_t first, size_t last) {
	// 	return net->forward(get_batch(first, last));
	// }
	

	Output initial_forward(StateType &s)
	{
		auto dts = device_tensor(s, net->device);
		auto rdts = torch::reshape(dts, {1, STATE_SIZE});
		return net->initial_forward(rdts);
	}

	Output recurrent_forward(StateType &s, ActionType &a)
	{
		auto dts = device_tensor(s, net->device);
		auto rdts = torch::reshape(dts, {1, STATE_SIZE});
		return net->recurrent_forward(rdts, a);
	}

	ActionType get_action(StateType &s, int simulations = 25, std::string id = "")
	{
		MCTSNode root;
		root.init_root(*this, s);
		MCTSNode::params p(GAMMA);
		for(int i = 0; i < simulations; ++i)
		{
			root.one_simulation(*this, p);
		}
		if(rand() % 100 == 0)
		{
			root.to_dot(std::string("mcts/") + id + std::string("_") + std::to_string(n_saved_trees));
			++n_saved_trees;
		}
		size_t action_idx = root.sample_action(100.0f);
		ActionType a;
		a.fill(0);
		a[action_idx] = 1;
		return a;
	}

	void record_action(StateType &s, ActionType a, float immidiate_reward, float value)
	{
		states.push_back(s);
		actions.push_back(a);

		one_hot_actions.emplace_back(std::array<float, ACTION_SIZE>{});
		for (size_t i = 0; i < ACTION_SIZE; ++i)
		{
			if (a[i])
			{
				one_hot_actions.back()[i] = 1.0;
			}
			else
			{
				one_hot_actions.back()[i] = 0.0;
			}
		}
		immidiate_rewards.push_back(immidiate_reward);
		values.push_back(value);
	}

	void append_experience(Model &m)
	{
		states.insert(states.end(), m.states.begin(), m.states.end());
		actions.insert(actions.end(), m.actions.begin(), m.actions.end());
		one_hot_actions.insert(one_hot_actions.end(), m.one_hot_actions.begin(), m.one_hot_actions.end());
		immidiate_rewards.insert(immidiate_rewards.end(), m.immidiate_rewards.begin(), m.immidiate_rewards.end());
		rewards.insert(rewards.end(), m.rewards.begin(), m.rewards.end());
		normalized_rewards.insert(normalized_rewards.end(), m.normalized_rewards.begin(), m.normalized_rewards.end());
		values.insert(values.end(), m.values.begin(), m.values.end());
		adv.insert(adv.end(), m.adv.begin(), m.adv.end());
	}

	void remove_frame(size_t frame)
	{
		actions.erase(actions.begin() + frame);
		one_hot_actions.erase(one_hot_actions.begin() + frame);
		states.erase(states.begin() + frame);
		time_stamps.erase(time_stamps.begin() + frame);
		immidiate_rewards.erase(immidiate_rewards.begin() + frame);
		values.erase(values.begin() + frame);
		rewards.erase(rewards.begin() + frame);
		normalized_rewards.erase(normalized_rewards.begin() + frame);
		adv.erase(adv.begin() + frame);
	}

	ActionType saved_action(int frame)
	{
		return actions[frame];
	}

	int get_frames()
	{
		return actions.size();
	}

	// Net<TAction, StateParam::count()> net;
	std::shared_ptr<Net> net;
	torch::optim::Adam optimizer;
	//torch::optim::Adam value_optimizer;
	std::vector<ActionType> actions;
	std::vector<std::array<float, ACTION_SIZE>> one_hot_actions;
	std::vector<StateType> states;
	std::vector<uint32_t> time_stamps;
	std::vector<float> immidiate_rewards;
	std::vector<float> rewards;
	std::vector<float> normalized_rewards;
	std::vector<float> adv;
	std::vector<float> values;
	int n_saved_trees;
};