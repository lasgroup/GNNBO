from graph_env.graph_generator import generate_graph_data
from graph_env.kernel_function import GntkJax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from typing import Optional

class Reward:
    def __init__(self, evidence_size: int, num_nodes: int, dim_feats: int, num_mlp_layers: int, noise_var: float,
                 sampling_mode: str = 'binomial', edge_prob: Optional[float] = None, num_edges: Optional[int] = None,
                 random_state = None, evidence_reward: Optional[jnp.ndarray] = None, random_evidence: str = 'prior',
                 ):
        self._rds = np.random if random_state is None else random_state
        self.noise_var = noise_var
        self.num_nodes = num_nodes
        self.feat_dim = dim_feats
        # prepare the evidence to calculate the posterior based on
        self.graph_evidence = generate_graph_data(data_size=evidence_size, num_nodes = num_nodes, dim_feats=dim_feats,
                                             sampling_mode = sampling_mode, edge_prob=edge_prob, num_edges = num_edges, random_state=self._rds)

        # generate the kernel matrix
        self.kernel = GntkJax(num_nodes=num_nodes, num_mlp_layers=num_mlp_layers)

        #prepare the prior
        # https://sandipanweb.wordpress.com/2020/12/08/gaussian-process-regression-with-python/
        self.gram_ev = self.kernel.get_kernel_matrix(self.graph_evidence)
        self.phi_ev = jnp.linalg.cholesky(self.gram_ev + self.noise_var * jnp.eye(evidence_size))

        # for now I'm writing this class for one reward.
        num_sample_funcs = 1
        # If we need speed up, the class can be altered s.t. it spits out multiple rewards functions with roughly same computation
        if evidence_reward is not None:
            assert evidence_reward.shape == (evidence_size, num_sample_funcs)
            self.y = evidence_reward
        else:
            if random_evidence == 'prior':
                #draw some random coefficients
                beta = self._rds.normal(size=(evidence_size, num_sample_funcs))
                f_prior = jnp.dot(self.phi_ev, beta)
                self.y = f_prior + self._rds.normal(0,self.noise_var, size = f_prior.shape) #can be just random
            if random_evidence == 'random':
                self.y = self._rds.normal(0,1,size = (evidence_size, num_sample_funcs))

        self.true_labels = []
        self.max_val = -1

    # def graph_sine(self, graph_list):
    #     w = self._rds.normal(size=(self.num_nodes, self.feat_dim))
    #     a = [jnp.sin(jnp.sum(w*graph.feat_mat_aggr_normed())) for graph in graph_list]
    #     return np.array(a)

    def generate_dataset(self, domain: list):
        # find the minimum value of f on a given domain
        self.true_labels = self.rew_func(domain)
        self.max_val = jnp.max(np.array(self.true_labels))

    def evaluate(self, idx_list: list, noisy: bool):
        rew = jnp.array([self.true_labels[idx] for idx in idx_list])
        if noisy:
            rew = rew + self._rds.normal(0, self.noise_var, size = rew.shape)
        return list(rew)

    def rew_func(self, graph_test: list):
        # graph_test should be a list of graphs. Could be a list of length 1.
        gram_cross = self.kernel.get_kernel_matrix(self.graph_evidence,graph_test)
        phi_cross = jnp.linalg.solve(self.phi_ev, gram_cross)
        mu = jnp.dot(phi_cross.T, jnp.linalg.solve(self.phi_ev, self.y))
        # we set the reward as the posterior mean.


        # # for drawing a random function from the posterior distribution:
        # gram_test = self.kernel.get_kernel_matrix(graph_test)
        # phi_test = np.linalg.cholesky(gram_test + self.noise_var*np.eye(len(graph_test)) - np.dot(phi_cross.T, phi_cross))
        # num_sample_funcs = 1
        # f_post = mu.reshape(-1,1) + np.dot(phi_test, self._rds.normal(size=(len(graph_test),num_sample_funcs)))

        # # for posterior variance do:
        # gram_test = self.kernel.get_kernel_matrix(graph_test)
        # post_sd = np.sqrt(np.diag(gram_test) - np.sum(phi_cross**2, axis = 0))
        return list(mu)

        #return list(f_post)

if __name__ == '__main__':
    num_nodes = 6
    dim_feats = 5
    sampling_mode = 'binomial'
    edge_prob = 0.9
    data_size = 5
    test_size = 20
    noise_var = 0.01
    num_mlp_layers = 3

    t0 = time.time()
    rds = np.random.RandomState(800)
    reward = Reward(evidence_size=data_size, num_nodes=num_nodes, dim_feats=dim_feats, num_mlp_layers=num_mlp_layers,
                    noise_var=noise_var, edge_prob=edge_prob, random_state=rds)

    graph_test = generate_graph_data(data_size=test_size, num_nodes = num_nodes, dim_feats=dim_feats, sampling_mode = sampling_mode, edge_prob=edge_prob, random_state=rds)
    print('generating reward class and sampling graphs:', (time.time() - t0) / 60)
    t0 = time.time()
    reward.generate_dataset(graph_test)
    indices =list(jnp.arange(test_size))
    labels = reward.evaluate(indices, noisy = False)

    plt.plot(labels)
    plt.show()
    print('calculating reward for the new graphs:', (time.time()-t0)/60)
