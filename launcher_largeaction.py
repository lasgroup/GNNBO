from utils_exp import generate_base_command, generate_run_commands, hash_dict
from config import RESULT_DIR, alg_lambdas, alg_betas, alg_pretrain_steps, alg_intersect
import run_gnnucb, run_phasedgp
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'GNN-UCB':['exploration_coef','pretrain_steps', 'alg_lambda','neuron_per_layer','net', 't_intersect'],
    'Dataset': ['num_nodes','feat_dim','edge_prob', 'num_actions']
}

default_configs = {
    # Dataset
    'num_nodes': 20, # or 20 or 100
    'edge_prob': 0.2, #or 0.2 or 0.95
    'feat_dim': 10, # 10 or 100
    'num_actions': 1000, # any number below 10000 works.
    # GNN-UCB
    'pretrain_steps': alg_pretrain_steps['GNN_US'][0],
    'neuron_per_layer': 2048,
    'exploration_coef': alg_betas['GNN_US'][0],
    'alg_lambda': alg_lambdas['GNN_US'][0],
    't_intersect': alg_intersect['GNN_US'][0],
    'net': 'GNN'
}

search_ranges = {

}


# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}

def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10**rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    elif sample_type == 'intuniform':
        return rds.randint(*range)
    else:
        raise NotImplementedError

def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 101
    init_seeds = list(rds.randint(0, 10**6, size=(101,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    #exp_path = os.path.join(exp_base_path, '%s'%(args.net))
    exp_path = exp_base_path


    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'exp_name', 'num_cpus']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            cmd = generate_base_command(run_phasedgp, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    #generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='euler', promt=True, long=True, mem=4000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--exp_name', type=str,  default='large_actionset_new')
    parser.add_argument('--num_cpus', type=int, default=8)
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=10)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--data', type=str, default='synthetic_data', help='dataset type')
    parser.add_argument('--seed', type=int, default=864, help='random number generator seed')
    #parser.add_argument('--runner_verbose', type=bool, default=False)

    # model arguments
    # this is to set algo params that you don't often want to change
    #parser.add_argument('--net', type=str, default='GNN', help='Network to use for UCB')
    parser.add_argument('--nn_aggr_feat', type=bool, default=True)
    parser.add_argument('--nn_init_lazy', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--T', type=int, default=1000) #change to 1500
    parser.add_argument('--T0', type=int, default=150)

    args = parser.parse_args()
    main(args)