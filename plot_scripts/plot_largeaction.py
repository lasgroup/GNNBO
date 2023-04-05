from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plot_specs import *
import bundles
from config import alg_lambdas, alg_betas, alg_pretrain_steps

DIR = '/local/pkassraie/gnnucb/results/'

# set the experiment setting you would like to see:
configs = {
    # Dataset
    'feat_dim': 10, #or 10
    'num_actions': 1000,
    'num_nodes': 20,
    'edge_prob': 0.2,
    # other BO params
    'algo': '_UCB',
    'T': 1000
}


plt.rcParams.update(bundles.neurips2022(ncols = 3, tight_layout=True))
plot_name = 'regret_{}p_{}d_{}N_labeled'.format(configs['edge_prob'], configs['feat_dim'], configs['num_nodes'])
fig, axes = plt.subplots(ncols = 3)
plt.rcParams.update(bundles.neurips2022(ncols = 3, tight_layout=True))

for num_actions, T, col_counter in zip([500, 1000], [500,1000], [1,2]):
    for net in [ 'GNN','NN']:
        for policy in ['us', 'ucb']:
            print('Trying:', policy, net, num_actions, T)
            if policy == 'us' and net == 'GNN':
                df_full, _ = collect_exp_results(exp_name='large_actionset_new')
            else:
                df_full, _ = collect_exp_results(exp_name='large_actionset')

            algo = net + '_' + policy.upper()
            df = df_full.loc[df_full['num_actions']==num_actions]
            df = df.loc[df['feat_dim'] == configs['feat_dim']]
            df = df.loc[df['num_nodes'] == configs['num_nodes']]
            df = df.loc[df['edge_prob'] == configs['edge_prob']]
            df = df.loc[df['net'] == net]
            df = df.loc[df['algorithm'] == policy]
            label = r'$\vert \mathcal{G}\vert$ = '+ str(num_actions)
            #plot
            if algo == 'NN_UCB':
                df = df.loc[df['T'] == T]
            regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in df['regrets_bp']])

            regrets_bp = regrets_bp[:,0:T]
            axes[col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[algo],label = algo_names[algo], color =  line_color[algo])
            axes[col_counter].fill_between(np.arange(T), np.mean(regrets_bp, axis=0) - 0.1 * np.std(regrets_bp, axis=0),
                                              np.mean(regrets_bp, axis=0) + 0.1 * np.std(regrets_bp, axis=0), alpha=0.3,
                                              color =  line_color[algo])
            axes[col_counter].set_title(label)
            axes[col_counter].set_xlabel(r'$t$')

DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='scalability')
df_full_us, _ = collect_exp_results(exp_name='scalability_us_new')
# set the experiment setting you would like to see:
configs = {
    # Dataset
    'feat_dim': 10, #or 10
    'num_actions': 200,
    # other BO params
    'T' : 500,
    'num_nodes': 20,
    'edge_prob': 0.2,
}
# pick which datasets
df = df_full.loc[df_full['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]
df = df.loc[df['num_nodes'] == configs['num_nodes']]
df = df.loc[df['edge_prob'] == configs['edge_prob']]

df_us = df_full_us.loc[df_full_us['num_actions']==configs['num_actions']]
df_us = df_us.loc[df_us['feat_dim'] == configs['feat_dim']]
df_us = df_us.loc[df_us['T'] == configs['T']]
df_us = df_us.loc[df_us['num_nodes'] == configs['num_nodes']]
df_us = df_us.loc[df_us['edge_prob'] == configs['edge_prob']]

label = r'$\vert \mathcal{G}\vert$ = '+ str(configs['num_actions'])

for net in ['GNN', 'NN']:
    algo = net + '_UCB'
    algo_us = net + '_US'
    df_net = df.loc[df['net'] == net]
    df_net_us = df_us.loc[df_us['net'] == net]

    df_net_us = df_net_us.loc[df_net_us['alg_lambda'] == alg_lambdas[algo_us][0]]
    df_net_us = df_net_us.loc[df_net_us['exploration_coef'] == alg_betas[algo_us][0]]

    # plot
    regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in df_net['regrets_bp']])
    axes[0].plot(np.mean(regrets_bp, axis=0), linestyle[algo], label=algo_names[algo],
                 color=line_color[algo])
    axes[0].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.1 * np.std(regrets_bp, axis=0),
                         np.mean(regrets_bp, axis=0) + 0.1 * np.std(regrets_bp, axis=0), alpha=0.3,
                         color=line_color[algo])

    regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in df_net_us['regrets_bp']])
    axes[0].plot(np.mean(regrets_bp, axis=0), linestyle[algo_us], label=algo_names[algo_us], color=line_color[algo_us])
    axes[0].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.1 * np.std(regrets_bp, axis=0),
                         np.mean(regrets_bp, axis=0) + 0.1 * np.std(regrets_bp, axis=0), alpha=0.3,
                         color=line_color[algo_us])
    axes[0].set_title(label)
axes[0].set_ylabel(r'$\hat{R}_t$')
axes[0].set_xlabel(r'$t$')
lines_labels = [ax.get_legend_handles_labels() for ax in [axes[1]]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, -0.02),fancybox =True, ncol=4)



plt.rcParams.update(bundles.neurips2022(ncols = 3, tight_layout=True))
plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

plt.show()