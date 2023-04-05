from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plot_specs import *
import bundles
DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='hyper_param_search_ucb')

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
    'T' : 400
}

# pick which synthetic
df = df_full.loc[df_full['num_nodes'] == configs['num_nodes']]
df = df.loc[df['edge_prob'] == configs['edge_prob']]
df = df.loc[df['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]

plot_name = 'new_paper_hyperparam_{}T_{}N_{}d_{}p_{}actions'.format(configs['T'],configs['num_nodes'],configs['feat_dim'],configs['edge_prob'], configs['num_actions'])
plt.rcParams.update(bundles.neurips2022(ncols=2,nrows=2,  tight_layout=True))
fig, axes = plt.subplots( ncols = 2, nrows=2)#, figsize = (8, 12)
plt.rcParams.update(bundles.neurips2022(ncols=2,nrows=2,  tight_layout=True))

row_counter = 0
col_counter =0
for net in ['GNN', 'NN']:
    counter = 0
    df_net = df.loc[df['net'] == net]
    configurations = [config for config in
                      zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if
                      not all(z == config[0] for z in config[1:])]
    configurations = list(set(configurations))
    # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
    for config in configurations:
        # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
        alg_lambda = config[0]
        exp_coef = config[1]
        pretrain_steps = config[2]
        sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
        sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]

        curve_name = r'$\lambda = $' + '{:.5f}'.format(alg_lambda) + r'$- \beta = $' + '{:.5f}'.format(exp_coef)
        # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
        regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
        # picked_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['pick_vars_all']])
        # avg_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['avg_vars']])

        axes[row_counter][col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = generic_lines[counter])
        axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.2*np.std(regrets_bp, axis=0),
                             np.mean(regrets_bp, axis=0)+0.2*np.std(regrets_bp, axis=0), alpha=0.2,
                             color = generic_lines[counter])

        counter += 1

    #axes[row_counter][col_counter].set_xlabel(r'$t$')
    axes[row_counter][col_counter].set_title(f'{net}-UCB')
    #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
    col_counter += 1

DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='hyper_param_search_us_new')
row_counter += 1
col_counter = 0
# # pick which synthetic
df = df_full.loc[df_full['num_nodes'] == configs['num_nodes']]
df = df.loc[df['edge_prob'] == configs['edge_prob']]
df = df.loc[df['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]

for net in ['GNN', 'NN']:
    counter = 0
    df_net = df.loc[df['net'] == net]
    if net == 'NN':
        df_net = df_net.loc[df_net['pretrain_steps']!= 40]
    configurations = [config for config in
                      zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if
                      not all(z == config[0] for z in config[1:])]
    configurations = list(set(configurations))
        # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
    for config in configurations:
        #for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
        alg_lambda = config[0]
        exp_coef = config[1]
        pretrain_steps = config[2]
        sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
        sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
        sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]

        curve_name = r'$\lambda = $' + '{:.5f}'.format(alg_lambda) + r'$- \beta = $' + '{:.5f}'.format(exp_coef)
        # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
        regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
        # picked_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['pick_vars_all']])
        # avg_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['avg_vars']])
        axes[row_counter][col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_US'], label=curve_name)#, color=generic_lines_us[counter])
        axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.2 * np.std(regrets_bp, axis=0),
                                       np.mean(regrets_bp, axis=0) + 0.2 * np.std(regrets_bp, axis=0), alpha=0.2,)
                                           #color=generic_lines_us[counter])
        # axes[row_counter][col_counter].plot(np.mean(regrets, axis=0), linestyle[net], label=curve_name, color=generic_lines_gp[counter])
        # axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(regrets, axis=0) - np.std(regrets, axis=0),
        #                         np.mean(regrets, axis=0) + np.std(regrets, axis=0), alpha=0.2,
        #                         color=generic_lines_gp[counter])
        counter += 1

    #axes[row_counter][col_counter].set_xlabel(r'$t$')

    axes[row_counter][col_counter].set_title(net+'-PE')
    #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
    col_counter += 1


lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[0][1],axes[1][0],axes[1][1]]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines,labels, loc ='center',bbox_to_anchor=(0.5, -0.4), fancybox = True, ncol = 4)

#fig.tight_layout()
plt.rcParams.update(bundles.neurips2022(ncols=2, nrows = 2,  tight_layout=True))
plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

#plt.show(bbox_inches='tight')