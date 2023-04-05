from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
from plot_specs import *
import bundles
from config import alg_lambdas, alg_betas, alg_pretrain_steps
DIR = '/local/pkassraie/gnnucb/results/'
df_full, _ = collect_exp_results(exp_name='scalability')
df_full_us, _ = collect_exp_results(exp_name='scalability_us')
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

#plt.rcParams.update(bundles.neurips2022(ncols=3,nrows =4, tight_layout=True))
plt.rcParams.update(bundles.neurips2022(ncols=2, tight_layout=True))
plot_name = 'paper_density2_all_{}d_labeled'.format(configs['feat_dim'])
fig, axes = plt.subplots( ncols=2)
plt.rcParams.update(bundles.neurips2022(ncols=2, tight_layout=True))

generic_lines = ['#949494','#FBAFE4','#ff6961']
generic_lines.reverse()

col_counter = 0
for net in ['GNN', 'NN']:
    algo = net + '_UCB'
    algo_us = net + '_US'
    df_net = df.loc[df['net'] == net]
    df_net_us = df_us.loc[df_us['net'] == net]


    #for num_nodes in [ 100]:#N
    num_nodes = 100
    df_n = df_net.loc[df_net['num_nodes'] == num_nodes]
    df_n_us = df_net_us.loc[df_net_us['num_nodes'] == num_nodes]
    for color_counter, edge_prob in enumerate([0.05, 0.2, 0.95]):
        df_p = df_n.loc[df_n['edge_prob'] == edge_prob]
        df_p_us = df_n_us.loc[df_n_us['edge_prob'] == edge_prob]

        # df_net = df_net.loc[df_net['alg_lambda'] == alg_lambdas[algo][d_counter]]
        # def_net = df_net.loc[df_net['exploration_coef'] == alg_betas[algo][d_counter]]
        # def_net = df_net.loc[df_net['pretrain_steps'] == alg_pretrain_steps[algo][d_counter]]

        # def_net_us = df_net_us.loc[df_net_us['alg_lambda'] == alg_lambdas[algo_us][d_counter]]
        # def_net_us = df_net_us.loc[df_net_us['exploration_coef'] == alg_betas[algo_us][d_counter]]
        # def_net_us = df_net_us.loc[df_net_us['pretrain_steps'] == alg_pretrain_steps[algo_us][d_counter]]
        curve_name = r' $p = $' + str(edge_prob)
        regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in df_p['regrets_bp']])
        regrets_bp_us = np.array([np.squeeze(np.array(regrets)) for regrets in df_p_us['regrets_bp']])

        axes[col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[algo], label = curve_name, color = generic_lines[color_counter], linewidth = 2)
        axes[col_counter].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.1 * np.std(regrets_bp, axis=0),
                                          np.mean(regrets_bp, axis=0) + 0.1 * np.std(regrets_bp, axis=0), alpha=0.3,
                                          color = generic_lines[color_counter], linewidth = 2)

        # axes[col_counter+1].plot(np.mean(regrets_bp_us, axis=0), linestyle[algo_us], label = curve_name, color = generic_lines[color_counter])
        # axes[col_counter+1].fill_between(np.arange(configs['T']), np.mean(regrets_bp_us, axis=0) - 0.1 * np.std(regrets_bp_us, axis=0),
        #                                   np.mean(regrets_bp_us, axis=0) + 0.1 * np.std(regrets_bp_us, axis=0), alpha=0.3,
        #                                   color = generic_lines[color_counter])

        axes[col_counter].set_xlabel(r'$t$')
        axes[col_counter].set_title( algo_names[algo])#+  r', $ N = $' + str(num_nodes))
        # axes[col_counter+1].set_title(algo_names[algo_us])# + r', $ N = $' + str(num_nodes))
        #axes[col_counter].set_ylabel(r'$R_t$')


    col_counter += 1


axes[0].set_ylabel(r'$\hat{R}_t$')
lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0]]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, -0.02),fancybox =True, ncol=4,prop={'size': 10})

#plt.rcParams.update(bundles.neurips2022(ncols=2, tight_layout=True))
plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

plt.show()