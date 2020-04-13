#include "model.h"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include "dirichlet.h"

MCTSNode::MCTSNode() : N(0U), R(0.0f), Q(0.0f)
{
}

void MCTSNode::to_dot(std::string out_file)
{
    std::string out = "digraph D {";
    out += "label=\"";
    for(auto p: P) {
        out += std::to_string(p) + std::string(",");
    }
    out += "\"\n";
    out += to_dot_internal(0, 0);
    out += "}";
    std::ofstream f(out_file);
    f << out;
    f.close();
}

std::string MCTSNode::to_dot_internal(int idx, int depth)
{
    auto to_id = [](int idx, int depth) -> std::string { return std::string("d") + std::to_string(depth) + std::string("i") + std::to_string(idx); };
    std::string out;
    out += to_id(idx, depth) + std::string("[xlabel=\"") + std::to_string(N) + std::string("\"]\n");
    for(int i = 0; i < children.size(); ++i)
    {
        if(children[i]->N > 0) {
            out += to_id(idx, depth) + std::string(" -> ") + to_id(i, depth + 1) + std::string("\n");
            out += children[i]->to_dot_internal(i, depth + 1);
        }
    }
    return out;
}

void MCTSNode::init_root(Model &m, StateType &s)
{
    populate(m.initial_forward(s));
    // std::cout << "root: [";
    for(int i = 0; i < ACTION_SIZE; ++i)
    {
        // std::cout << P[i] << ", ";
    }
    // std::cout << "]" << std::endl;
    size_t idx = rand() % dirichlet.size();
    float frac = 0.25;
    for(int i = 0; i < ACTION_SIZE; ++i)
    {
        P[i] = (1 - frac) * P[i] + frac * dirichlet[idx][i];
    }
    // N = 1;
}
void MCTSNode::populate(Output o)
{
    hidden = o.hidden;
    Q = o.value.item<float>();
    R = o.reward.item<float>();
    auto dist = torch::softmax(o.policy, 1);
    auto dist_a = dist.accessor<float, 2>();
    children.resize(ACTION_SIZE);
    for(size_t i = 0; i < ACTION_SIZE; ++i)
    {
        children[i] = std::make_unique<MCTSNode>();
        P[i] = dist_a[0][i];
    }
}

void MCTSNode::one_simulation(Model &m, params &p) {
    p.path.clear();
    step(m, p);
    update_statistics(p);
}
void MCTSNode::step(Model &m, params &p) {
    constexpr float c1 = 1.25f;
    constexpr float c2 = 19625.0f;
    float max_vinst = -INFINITY;
    float Nsum = N;//nsum();
    size_t max_i = 5;
    std::array<size_t, ACTION_SIZE> perm;
    for(int i = 0; i < ACTION_SIZE; ++i)
    {
        perm[i] = i;
    }
    std::random_shuffle(perm.begin(), perm.end());
    // if(p.path.size() == 0)
    //     std::cout << "root step: [";
    for(size_t j = 0; j < children.size(); ++j) {
        size_t i = perm[j];
        auto &n = children[i];
        float Q = n->Q;
        if(p.qmax > p.qmin) {
            Q = (n->Q - p.qmin) / (p.qmax - p.qmin);
        }
        // std::cout << n->R + Q << ", ";
        float vinst = n->R + Q + P[i] * sqrt(Nsum) / (1.0f + n->N) * (c1 + log((Nsum + c2 + 1) / c2));
        // if(p.path.size() == 0)
        //     std::cout << vinst << ", ";
        if(vinst > max_vinst) {
            max_i = i;
            max_vinst = vinst;
        }
    }
    // if(p.path.size() == 0)
    //     std::cout << "]" << std::endl;
    // if(p.path.size() == 0)
    //     std::cout << "maxQ: " << max_i << std::endl;
    // std::cout << std::endl;
    p.path.push_back(this);
    if(children[max_i]->children.empty())
    {
        ActionType max_action;
        max_action.fill(0);
        max_action[max_i] = 1;
        children[max_i]->populate(m.net->recurrent_forward(hidden, max_action));
        p.path.push_back(children[max_i].get());
    }
    else
    {
        children[max_i]->step(m, p);
    }
}
float MCTSNode::nsum() const {
    float Nsum = 0;
    for(auto &it: children) {
        Nsum += it->N;
    }
    return Nsum;
}
	// children[action_idx] = 
void MCTSNode::update_statistics(params &p)
{
    auto &path = p.path;
    int l = (int)path.size();
    for(int k = (int)path.size() - 1; k >= 0; --k)
    {
        float Gk = pow(p.gamma, l - k) * path[l - 1]->Q;
        for(int tau = 0; tau < l - 1 - k; ++tau)
        {
            //####################################################################################################################
            Gk += pow(p.gamma, tau) * path[k + 1 + tau]->R;
        }
        path[k]->Q = (path[k]->N * path[k]->Q + Gk) / (path[k]->N + 1.0f);
        p.qmax = fmax(p.qmax, path[k]->Q);
        p.qmin = fmin(p.qmin, path[k]->Q);
        ++path[k]->N;
    }
}

size_t MCTSNode::sample_action(float T) const
{
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float Nsump = 0.0f; //pow(nsum(), 1.0 / T);
    for(int i = 0; i < ACTION_SIZE; ++i)
    {
        Nsump += pow(children[i]->N, 1.0 / T);
    }
    std::array<float, ACTION_SIZE> distribution;
    std::transform(children.begin(), children.end(), distribution.begin(),
        [T, Nsump](auto &child) -> float {
            return pow(child->N, 1.0 / T) / Nsump; 
    });
    float sum = 0;
    // std::cout << "action" << std::endl;
    // std::cout << "dist: [";
    // for(int i = 0; i < ACTION_SIZE; ++i)
    // {
    //     std::cout << distribution[i] << ", ";
    // }
    // std::cout << "]" << std::endl;
    // r = 0.85
    // [0.1, 0.2, 0.5, 0.2]
    for(int i = 0; i < ACTION_SIZE; ++i)
    {
        if(r >= sum && r < sum + distribution[i])
        {
            // std::cout << "choosen: " << i << std::endl;
            return i;
        }
        sum += distribution[i];
    }
    return ACTION_SIZE - 1;
}
