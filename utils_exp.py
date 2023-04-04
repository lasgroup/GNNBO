import sys
import os
import json
import time

import numpy as np
import glob
import pandas as pd
from config import DATA_DIR, RESULT_DIR
from graph_env.graph_generator import Graph

""" Gather synthetic data """

def collect_dataset(args, verbose=True):
    exp_dir = os.path.join(DATA_DIR, args.data)
    no_results_counter = 0
    data_dicts = []
    param_names = set()
    data_counter = 0
    for results_file in glob.glob(exp_dir + '/*.json'):
        #print('Loading Dataset #', data_counter+1)
        data_counter += 1
        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    data_dict = json.load(f)
                if (data_dict['params']['num_nodes'] == args.num_nodes) and (data_dict['params']['edge_prob'] == args.edge_prob) and (data_dict['params']['feat_dim'] == args.feat_dim):
                    data_dicts.append({**data_dict['dataset'], **data_dict['params']})
                    param_names = param_names.union(set(data_dict['params'].keys()))
                    #print('Dataset Matches description!')
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            args.data, len(data_dicts), no_results_counter))

    return pd.DataFrame(data=data_dicts), list(param_names)

def dataset_to_graphdata(dataset):
    graph_features = list(dataset['features'])[0]
    graph_connections = list(dataset['connections'])[0]
    num_nodes = list(dataset['num_nodes'])[0]
    feat_dim = list(dataset['feat_dim'])[0]
    graph_data = [Graph(dim_feats=feat_dim, num_nodes=num_nodes,adj_mat=np.array(adj_mat),feat_mat=np.array(feat_mat) ) for adj_mat, feat_mat in zip(graph_connections,graph_features)]
    return graph_data

def read_dataset(args, env_rds):
    t0 = time.time()
    # Read the datasets
    datasets, _= collect_dataset(args)
    # pick one dataset that matches the environment setting
    datasets = datasets.loc[datasets['num_nodes'] == args.num_nodes]
    datasets = datasets.loc[datasets['edge_prob'] == args.edge_prob]
    datasets = datasets.loc[datasets['feat_dim'] == args.feat_dim]
    datasets = datasets.loc[datasets['noise_var'] == args.noise_var]
    datasets = datasets.loc[datasets['num_mlp_layers'] == args.num_mlp_layers]
    env_seed = env_rds.choice(datasets['env_seed'])
    datasets = datasets.loc[datasets['env_seed'] == env_seed]
    graph_rewards = list(datasets['rewards'])[0]
    graph_rewards = [item for sublist in graph_rewards for item in sublist]
    graph_data = dataset_to_graphdata(datasets)
    print('Loading data took:', (time.time()-t0)/60)
    return graph_data, graph_rewards

""" Gather exp results """

def collect_exp_results(exp_name, verbose=True):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0
    print(exp_dir)
    exp_dicts = []
    param_names = set()
    for results_file in glob.glob(exp_dir + '/*/*.json'): #might have to change the regex thing
        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                exp_dicts.append({**exp_dict['exp_results'], **exp_dict['params'], **{'algorithm': exp_dict['algorithm']}})
                param_names = param_names.union(set(exp_dict['params'].keys()))
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            exp_name, len(exp_dicts), no_results_counter))

    return pd.DataFrame(data=exp_dicts), list(param_names)

""" Async executer """
import multiprocessing

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass


""" Command generators """

def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd

def cmd_exec_fn(cmd):
    import os
    os.system(cmd)

def generate_run_commands(command_list, num_cpus=1, dry=False, n_hosts=1, mem=2000, long=False,
                          mode='local', promt=True, log_file_list=None):

    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'bsub ' + \
                   f'-W {23 if long else 3}:59 ' + \
                   f'-R "rusage[mem={mem}]" ' + \
                   f'-n {num_cpus} ' + \
                   f'-R "span[hosts={n_hosts}]" '

        if log_file_list is not None:
            assert len(command_list) == len(log_file_list)

        for python_cmd in command_list:
            if log_file_list is not None:
                log_file = log_file_list.pop()
                cluster_cmds.append(bsub_cmd + f'-o {log_file} -e {log_file} ' + python_cmd)
            else:
                cluster_cmds.append(bsub_cmd + python_cmd)

        if promt:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the euler cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local':
        if promt:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(f"About to launch {len(command_list)} commands in {num_cpus} local processes. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                exec.run(cmd_exec_fn, command_list)
    else:
        raise NotImplementedError

""" Hashing and Encoding dicts to JSON """

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def hash_dict(d):
    return str(abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))

if __name__ == '__main__':
    DIR = '/local/pkassraie/gnnucb/results/'
    df_full, _ = collect_exp_results(exp_name='testing_pipeline/NN')