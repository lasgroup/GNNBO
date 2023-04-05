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

method = 'us'

# pick which datasets
df = df_full.loc[df_full['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]
df_us = df_full_us.loc[df_full_us['num_actions']==configs['num_actions']]
df_us = df_us.loc[df_us['feat_dim'] == configs['feat_dim']]
df_us = df_us.loc[df_us['T'] == configs['T']]

plt.rcParams.update(bundles.neurips2022(ncols=3,nrows =3, tight_layout=True))
if method == 'us':
    plot_name = 'benchmark_variance_us_{}d'.format(configs['feat_dim'])
else:
    plot_name = 'benchmark_variance_ucb_{}d'.format(configs['feat_dim'])
fig, axes = plt.subplots( nrows =3,ncols=3)
plt.rcParams.update(bundles.neurips2022(nrows =3,ncols=3, tight_layout=True))


row_counter = 0
for num_nodes in [5, 20, 100]:

    df_n = df.loc[df['num_nodes'] == num_nodes]
    df_n_us = df_us.loc[df_us['num_nodes'] == num_nodes]
    col_counter = 0
    for edge_prob in [0.05, 0.2, 0.95]:

        df_p = df_n.loc[df_n['edge_prob'] == edge_prob]
        df_p_us = df_n_us.loc[df_n_us['edge_prob'] == edge_prob]
        for net in [ 'GNN','NN']:
            algo = net+'_UCB'
            algo_us = net+'_US'

            df_net = df_p.loc[df_p['net'] == net]
            df_net_us = df_p_us.loc[df_p_us['net'] == net]

            df_net_us = df_net_us.loc[df_net_us['alg_lambda'] == alg_lambdas[algo_us][d_counter]]
            df_net_us = df_net_us.loc[df_net_us['exploration_coef'] == alg_betas[algo_us][d_counter]]

            picked_vars = np.array([np.squeeze(np.array(vars)) for vars in df_net['pick_vars_all']])
            picked_vars_us = np.array([np.squeeze(np.array(vars)) for vars in df_net_us['pick_vars_all']])
            if method == 'us':
                axes[row_counter][col_counter].plot(np.mean(picked_vars_us, axis=0), label = algo_names[algo_us], color = line_color[algo_us])
                axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(picked_vars_us, axis=0) -  np.std(picked_vars_us, axis=0),
                                                  np.mean(picked_vars_us, axis=0) + np.std(picked_vars_us, axis=0), alpha=0.3,
                                                  color = shade_color[algo_us])
            else:
                axes[row_counter][col_counter].plot(np.mean(picked_vars, axis=0), label = algo_names[algo], color = line_color[algo])
                axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(picked_vars, axis=0) - np.std(picked_vars, axis=0),
                                                  np.mean(picked_vars, axis=0) + np.std(picked_vars, axis=0), alpha=0.3,
                                                  color = shade_color[algo])

        #axes[row_counter][col_counter].set_xlabel(r'$t$')
        axes[row_counter][col_counter].set_xlabel(r'$N = $' + str(num_nodes) +  r', $ p = $' + str(edge_prob))
        #axes[row_counter][col_counter].set_ylabel(r'$\sigma_t(G_t)$')


        col_counter += 1
    row_counter+= 1


lines_labels = [ax.get_legend_handles_labels() for ax in [axes[1][2]]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, -0.02),fancybox =True, ncol=4)

plt.rcParams.update(bundles.neurips2022(nrows =3,ncols=3, tight_layout=True))
plt.savefig(f'/local/pkassraie/gnnucb/plots/{plot_name}.pdf',bbox_inches='tight')

#plt.show()