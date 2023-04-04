from utils_exp import generate_base_command, generate_run_commands
from config import DATA_DIR
from graph_env import generate_dataset as generate_dataset
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'Dataset': ['num_nodes','feat_dim','edge_prob']
}

default_configs = {
    # Dataset
    'num_nodes': 100, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 100, # 10 or 100
}


# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys()}

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
    exp_base_path = os.path.join(DATA_DIR, args.exp_name)
    #exp_path = os.path.join(exp_base_path, '%s'%(args.net))
    exp_path = exp_base_path


    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'exp_name', 'num_cpus']]

        # randomly sample flags
        for flag in default_configs:
            flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        # flags_hash = hash_dict(flags)
        # flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            cmd = generate_base_command(generate_dataset, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    #generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='euler', promt=True, mem = 8000, long=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate data')
    # experiment parameters
    parser.add_argument('--exp_name', type=str,  default='synthetic_data')
    parser.add_argument('--num_cpus', type=int, default=8)
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=1)
    parser.add_argument('--seed', type=int, default=864, help='random number generator seed')

    # model arguments
    args = parser.parse_args()
    main(args)