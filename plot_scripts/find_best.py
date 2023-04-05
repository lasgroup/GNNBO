from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plot_specs import *
import bundles

configs = {
    # Dataset
    'num_nodes': 5, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 10, #10 or 100
    'num_actions': 200, # any number below 10000 works.
    # GNN-UCB
    'neuron_per_layer': 2048,
    'exploration_coef': 1e-3,
    'alg_lambda': 0.01,
    # other BO params
    'T' : 400,
    'net': 'NN'
}

plot_name = 'best_hyperparam_{}T_{}N_{}d_{}p_{}actions'.format(configs['T'],configs['num_nodes'],configs['feat_dim'],configs['edge_prob'], configs['num_actions'])
plt.rcParams.update(bundles.neurips2022())
fig = plt.figure()#, figsize = (8, 12)
plt.rcParams.update(bundles.neurips2022())

DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='hyper_param_search_us_new')

regret = 100
# pick which synthetic
df = df_full.loc[df_full['num_nodes'] == configs['num_nodes']]
df = df.loc[df['edge_prob'] == configs['edge_prob']]
df = df.loc[df['num_actions'] == configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]
df_net = df.loc[df['net'] == configs['net']]
count = 0
configurations = [config for config in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps'], df_net['t_intersect']) if not all(z == config[0] for z in config[1:])]
configurations = list(set(configurations))
regret_min = 100
best_lambda = 0
best_pretrain = 0
best_beta = 0
best_intersect = 0
#for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
for config in configurations:
    alg_lambda = config[0]
    exp_coef = config[1]
    pretrain_steps = config[2]
    t_intersect = config[3]
    sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
    sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
    sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]
    sub_df = sub_df.loc[sub_df['t_intersect'] == t_intersect]
    count += 1
    curve_name = r'$\lambda = $' + '{:.5f}'.format(alg_lambda) + r'$- \beta = $' + '{:.5f}'.format(exp_coef)
    # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
    regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
    regret_last = np.mean(regrets_bp, axis=0)[-1]
    plt.plot(np.mean(regrets_bp, axis=0), linestyle[configs['net']+'_US'])#, label=curve_name)
    plt.fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.2 * np.std(regrets_bp, axis=0),
                                   np.mean(regrets_bp, axis=0) + 0.2 * np.std(regrets_bp, axis=0), alpha=0.2,)
    if regret_last < regret_min:
        regret_min = regret_last
        best_lambda = alg_lambda
        best_beta = exp_coef
        best_pretrain = pretrain_steps
        best_intersect = t_intersect
print(regret_min)
print(best_lambda)
print(best_beta)
print(best_pretrain)
print(best_intersect)
#
# axes.set_title(net+'-PE')

#
# lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[0][1],axes[1][0],axes[1][1]]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# plt.legend(lines,labels, loc ='center',bbox_to_anchor=(0.5, -0.2), fancybox = True, ncol = 4)

#fig.tight_layout()
plt.rcParams.update(bundles.neurips2022())
#plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

plt.show()
