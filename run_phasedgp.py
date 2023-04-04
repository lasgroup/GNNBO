import os
import json
from algorithms import  PhasedGnnUCB
import time
import numpy as np
import torch
from utils_exp import NumpyArrayEncoder, read_dataset
from plot_scripts.utils_plot import plt_regret
from matplotlib import pyplot as plt
import argparse



def evaluate(idx_list: list, reward_list: list, noisy: bool, _rds , noise_var: float):
    rew = np.array([reward_list[idx] for idx in idx_list])
    if noisy:
        rew = rew + _rds.normal(0, noise_var, size=rew.shape)
    return list(rew)

def main(args):

    # read full data
    env_rds = np.random.RandomState(args.seed)
    graph_data_full, graph_rewards_full = read_dataset(args,env_rds)

    # Pick the data: The entire dataset has 10,000 graphs. Pick a random set of points to work with.
    indices = env_rds.choice(range(len(graph_data_full)), args.num_actions)

    graph_data = [graph_data_full[i] for i in indices]
    graph_rewards = [graph_rewards_full[i] for i in indices]

    max_reward = np.max(graph_rewards)
    max_graph = np.argmax(graph_rewards)
    # set bandit algorithm
    assert args.num_actions == len(graph_data)
    assert len(graph_data) == len(graph_rewards)
    algo_rds = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    learner = PhasedGnnUCB(net = args.net, feat_dim = args.feat_dim, num_nodes = args.num_nodes,num_actions = args.num_actions, action_domain = graph_data, verbose=args.runner_verbose,
                     alg_lambda = args.alg_lambda, exploration_coef = args.exploration_coef, t_intersect=args.t_intersect,
                   train_from_scratch=args.train_from_scratch, nn_aggr_feat=args.nn_aggr_feat,
                     num_mlp_layers = args.num_mlp_layers_alg, neuron_per_layer = args.neuron_per_layer, lr = args.lr, nn_init_lazy=args.nn_init_lazy, random_state=algo_rds)

    t0 = time.time()

    # run bandit algorithm
    regrets = []
    regrets_bp = []
    cumulative_regret = 0
    cumulative_regret_bp = 0
    new_indices = []
    new_rewards = []
    actions_all = []
    avg_vars = []
    pick_vars_all = []
    pick_rewards_all = []

    print(args.__dict__)
    for t in range(args.T):
        # only maximize ucb if you are passed pretrain time
        if t > args.pretrain_steps:
            action_t = learner.select()
        else: #otherwise, explore!
            action_t = learner.explore()

        actions_all.append(action_t)
        observed_reward_t = evaluate(idx_list = [action_t], noisy=args.noisy_reward, reward_list=graph_rewards, noise_var=args.noise_var, _rds = env_rds)
        pick_rewards_all.append(observed_reward_t)
        regret_t = max_reward - graph_rewards[action_t] #average (over noise) regret
        if args.runner_verbose:
            print('Maximum included in M_t:', max_graph in learner.maximizers)
            print('Lenth of maximizers:', len(learner.maximizers))
        cumulative_regret += regret_t
        #BP regret:
        best_action_t = learner.exploit()
        regret_t_bp = max_reward - graph_rewards[best_action_t]
        cumulative_regret_bp += regret_t_bp

        if t < args.T0:
            learner.add_data([action_t], [observed_reward_t])
            # only train the network if you are passed pretrain time
            if t > args.pretrain_steps:
                loss = learner.train()
        else:  # After some time just train in batches
            # save the new datapoints
            if len(new_rewards) > 0:
                new_rewards.append(observed_reward_t)
                new_indices.append(action_t)
            else:
                new_rewards = [observed_reward_t]
                new_indices = [action_t]
            # when there's enough, update the GP
            if t % args.batch_size == 0:
                learner.add_data(new_indices, new_rewards)
                # only train the network if you are passed pretrain time
                if t > args.pretrain_steps:
                    loss = learner.train()
                new_indices = []  # remove from unused points
                new_rewards = []
                #plot mean and variance estimates
        regrets.append(cumulative_regret)
        regrets_bp.append(cumulative_regret_bp)
        pick_vars_all.append(learner.get_post_var(action_t))

        if t % args.print_every == 0:
            if args.runner_verbose:
                print('Verbose is true')
                print('At step {}: Action{}, Regret {}'.format(t + 1, action_t, cumulative_regret))
                # plot conf ests
                means = np.array([learner.get_post_mean(idx) for idx in range(args.num_actions)])
                vars = np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
                avg_vars.append(np.mean(vars))

                plt.plot(means, '-', label='means', color='#9dc0bc')
                plt.title(f'Confidence and mean Estimates, t = {t}')
                plt.fill_between(np.arange(args.num_actions), means - np.sqrt(args.exploration_coef) * vars,
                                 means + np.sqrt(args.exploration_coef) * vars, alpha=0.2, color='#b2edc5')
                plt.plot(graph_rewards, label='true function', color='#7c7287')
                color = [item * 255 / (t + 1) for item in np.arange(t + 1)]
                plt.scatter(actions_all,
                            evaluate(idx_list=actions_all, noisy=False, reward_list=graph_rewards, noise_var=args.noise_var,
                                     _rds=env_rds), c=color)
                plt.set_cmap('magma')
                plt.legend()
                plt.show()
                if t > 0:
                    plt_regret(regrets = regrets, regrets_bp = regrets_bp,net = args.net, t=t, print_every=args.print_every,plot_vars=True,avg_vars=avg_vars, pick_vars_all=pick_vars_all)
    if args.runner_verbose:
        print(f'{learner.name} with {args.T} steps takes {(time.time() - t0)/60} mins.')
    exp_results = {'actions': actions_all, 'rewards': pick_rewards_all, 'regrets': regrets, 'regrets_bp': regrets_bp, 'pick_vars_all': pick_vars_all, 'avg_vars':avg_vars}

    results_dict = {
        'exp_results': exp_results,
        'params': args.__dict__,
        'duration_total': (time.time() - t0)/60,
        'algorithm': 'us'
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        exp_hash = str(abs(json.dumps(results_dict['params'], sort_keys=True).__hash__()))
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)
        print('Duration:', (time.time() - t0)/60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phased-GNN run')

    # environment arguments
    # this is to set which dataset to pick
    parser.add_argument('--data', type=str, default='easy_data_for_tests', help='dataset type')
    parser.add_argument('--num_nodes', type=int, default=5, help = 'max number of nodes per graph')
    parser.add_argument('--feat_dim', type = int, default=10, help ='Dimension of node features for the graph')
    parser.add_argument('--edge_prob', type=float, default=0.05, help='probability of existence of each edge, shows sparsity of the graph')
    parser.add_argument('--data_size', type=int, default=5, help = 'size of the seed dataset for generating the reward function')
    parser.add_argument('--num_actions', type=int, default=200, help = 'size of the actions set, i.e. total number of graphs')
    parser.add_argument('--noise_var', type=float, default=0.0001, help = 'variance of noise for observing the reward, if exists')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help = 'number of MLP layer for the GNTK that creates the synthetic data')
    parser.add_argument('--seed', type=int, default=384)
    parser.add_argument('--nn_init_lazy', type=bool, default=True)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--print_every', type=str, default=20)
    parser.add_argument('--runner_verbose', type=bool, default=True)

    # model arguments
    parser.add_argument('--net', type=str, default='GNN', help='Network to use for UCB')
    parser.add_argument('--noisy_reward', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_mlp_layers_alg', type=int, default=2)
    parser.add_argument('--train_from_scratch', type=bool, default=True)
    parser.add_argument('--pretrain_steps', type=int, default=40)
    parser.add_argument('--neuron_per_layer', type=int, default=2048)
    parser.add_argument('--exploration_coef', type=float, default=0.001341321712103193) #0.0098
    parser.add_argument('--alg_lambda', type=float, default=1.156e-03)
    parser.add_argument('--t_intersect', type = int, default= 80)
    parser.add_argument('--nn_aggr_feat', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--T', type=int, default=320)
    parser.add_argument('--T0', type=int, default=100)

    args = parser.parse_args()
    main(args)