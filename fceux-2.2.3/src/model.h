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

const int N_ACTIONS = 8U;
const int N_HIDDEN = 64U;
const int STATE_SIZE = 0x800;
const float ACTION_THRESHOLD = 0.5f;

enum class Action {
    UP = 0,
    DOWN,
    LEFT,
    RIGHT,
    A,
    B,
    START,
    SELECT,
    MAX
};

constexpr size_t ACTION_SIZE = static_cast<size_t>(Action::MAX);

typedef std::array<float, STATE_SIZE> StateType;
typedef std::array<uint8_t, ACTION_SIZE> ActionType;
struct Net : torch::nn::Module {
	Net() : device(get_device()) {
		bn = register_module("bn", torch::nn::BatchNorm(STATE_SIZE));
		fc1 = register_module("fc1", torch::nn::Linear(STATE_SIZE, N_HIDDEN));
		fc2 = register_module("fc2", torch::nn::Linear(N_HIDDEN, N_HIDDEN));
		fc3 = register_module("fc3", torch::nn::Linear(N_HIDDEN, N_ACTIONS));
		torch::nn::init::xavier_normal_(fc1->weight);
		torch::nn::init::xavier_normal_(fc2->weight);
		torch::nn::init::xavier_normal_(fc3->weight);
		torch::nn::init::zeros_(fc3->bias);
		torch::nn::init::zeros_(fc2->bias);
		torch::nn::init::zeros_(fc1->bias);
		to(device);
	}

	static torch::Device get_device() {
		if(getenv("NO_CUDA")) {
			return torch::kCPU;
		}
		if (torch::cuda::is_available()) {
			return torch::kCUDA;
		}
		else {
			return torch::kCPU;
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}

	torch::nn::BatchNorm bn{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	torch::Device device;
};

struct Model {
	typedef Net NetType;
	Model(float lr) : net{std::make_shared<NetType>()}, optimizer(net->parameters(), lr)
	{
	}

	friend std::ostream & operator<<(std::ostream &os, const Model &m) {
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
		std::stringstream ss;
		torch::save(net, ss);
		a(cereal::make_nvp("net", ss.str()));
		std::stringstream sso;
		torch::save(optimizer, sso);
		a(cereal::make_nvp("opt", sso.str()));
	}

	template <class Archive>
	void load(Archive &a)
	{
		a(cereal::make_nvp("actions", actions));
		a(cereal::make_nvp("states", states));
		a(cereal::make_nvp("immidiate_rewards", immidiate_rewards));
		std::string s;
		a(s);
		std::stringstream ss{s};
		torch::load(net, ss, net->device);		
		std::string so;
		a(so);
		std::stringstream sso{so};
		torch::load(optimizer, sso, net->device);		
	}

	torch::Tensor forward(StateType &s) {
		auto ts = torch::from_blob(static_cast<void*>(s.data()), {1, STATE_SIZE}, torch::kFloat32);
		auto dts = ts.to(net->device);
		return net->forward(dts);
	}

	torch::Tensor forward_batch_nice(size_t first, size_t last) {
		return net->forward(get_batch(first, last));
	}

	torch::Tensor get_batch(size_t first, size_t last) {
		auto ret = torch::from_blob(static_cast<void *>(states.data() + first), {static_cast<long>(last - first), STATE_SIZE}, torch::kFloat32);
		auto dret = ret.to(net->device);
		return dret;
	}

	torch::Tensor forward_batch_nice(torch::Tensor t) {
		return net->forward(t);
	}

	ActionType get_action(StateType &s) {
		auto tout = forward(s);
		torch::Tensor out = torch::sigmoid(tout).to(torch::kCPU);
		auto out_a = out.accessor<float,2>();
        std::array<uint8_t, ACTION_SIZE> actions;
        for(int i = 0; i < ACTION_SIZE; i++){
            actions[i] = out_a[0][i] > ACTION_THRESHOLD ? 1 : 0;
        }
		return actions;
	}

	void record_action(StateType &s, ActionType a, float immidiate_reward) {
		states.push_back(s);
		actions.push_back(a);

		one_hot_actions.emplace_back(std::array<float, ACTION_SIZE>{});
        for(int i = 0; i < ACTION_SIZE; ++i) {
            if(a[i]) {
		        one_hot_actions.back()[i] = 1.0;
            }
            else {
		        one_hot_actions.back()[i] = 0.0;
            }
        }
		immidiate_rewards.push_back(immidiate_reward);
	}

	ActionType saved_action(int frame) {
		return actions[frame];
	}

	int get_frames() {
		return actions.size();
	}

	// Net<TAction, StateParam::count()> net;
	std::shared_ptr<NetType> net;
	torch::optim::Adam optimizer;
	std::vector<ActionType> actions;
	std::vector<std::array<float, ACTION_SIZE>> one_hot_actions;
	std::vector<StateType> states;
	std::vector<uint32_t> time_stamps;
	std::vector<float> immidiate_rewards;
};