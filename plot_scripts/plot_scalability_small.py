from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plot_specs import *
import bundles
from config import alg_lambdas, alg_betas, alg_pretrain_steps
DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='scalability')
df_full_us, _ = collect_exp_results(exp_name='scalability_us_new')
# set the experiment setting you would like to see:
configs = {
    # Dataset
    'feat_dim': 10, #or 10
    'num_actions': 200,
    # other BO params
    'T' : 500
}
if configs['feat_dim'] == 10:
    d_counter = 0
else:
    d_counter = 1

# pick which datasets
df = df_full.loc[df_full['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]
df_us = df_full_us.loc[df_full_us['num_actions']==configs['num_actions']]
df_us = df_us.loc[df_us['feat_dim'] == configs['feat_dim']]
df_us = df_us.loc[df_us['T'] == configs['T']]

plt.rcParams.update(bundles.neurips2022(ncols=4, tight_layout=True))
plot_name = 'paper_scalability_{}d_labeled_hw'.format(configs['feat_dim'])
fig, axes = plt.subplots( ncols=4)
plt.rcParams.update(bundles.neurips2022(ncols=4, tight_layout=True))
col_counter =0

for net in ['GNN', 'NN']:
    algo = net + '_UCB'
    algo_us = net + '_US'

    df_net = df.loc[df['net'] == net]
    df_net_us = df_us.loc[df_us['net'] == net]


    i=0
    node_nums = [20, 100]
    for num_nodes in node_nums:
        df_n = df_net.loc[df_net['num_nodes'] == num_nodes]
        df_n_us = df_net_us.loc[df_net_us['num_nodes'] == num_nodes]

        df_p = df_n.loc[df_n['edge_prob'] == 0.2]
        df_p_us = df_n_us.loc[df_n_us['edge_prob'] == 0.2]

        label = r'$N=$'+str(num_nodes)
        regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in df_p['regrets_bp']])
        regrets_bp_us = np.array([np.squeeze(np.array(regrets)) for regrets in df_p_us['regrets_bp']])

        axes[col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[algo], label = label, color = generic_lines[i])
        axes[col_counter].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.1 * np.std(regrets_bp, axis=0),
                                          np.mean(regrets_bp, axis=0) + 0.1 * np.std(regrets_bp, axis=0), alpha=0.3,
                                          color = generic_lines[i])

        axes[col_counter+1].plot(np.mean(regrets_bp_us, axis=0), linestyle[algo_us], label = label, color = generic_lines[i])
        axes[col_counter+1].fill_between(np.arange(configs['T']), np.mean(regrets_bp_us, axis=0) - 0.1 * np.std(regrets_bp_us, axis=0),
                                          np.mean(regrets_bp_us, axis=0) + 0.1 * np.std(regrets_bp_us, axis=0), alpha=0.3,
                                          color = generic_lines[i])

        axes[col_counter].set_xlabel(r'$t$', fontsize=9)
        axes[col_counter].set_title( algo_names[algo])
        axes[col_counter+1].set_title( algo_names[algo_us])
        i+= 1

    col_counter += 2


lines_labels = [ax.get_legend_handles_labels() for ax in [axes[2]]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, -0.02),fancybox =True, ncol=2)

plt.rcParams.update(bundles.neurips2022(ncols=4, tight_layout=True))
plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

plt.show()