import math
import numpy as np
import scipy.sparse as sp
from graph_generator import Graph, generate_graph_data
from neural_tangents import stax
from jax.api import jit
import time
from typing import Optional

class GntkDu(object):
    """
    implement the Graph Neural Tangent Kernel
    """

    def __init__(self, num_block_ops, num_mlp_layers):
        """
        num_block_ops: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_block_ops = num_block_ops
        self.num_mlp_layers = num_mlp_layers

    @staticmethod
    def _next(cov_mat_s, diag1 = None, diag2 = None):
        """
        go through one normal layer, for all elements
        """
        if diag1 is None:
            diag1 =  np.sqrt(np.diag(cov_mat_s))
            diag2 =  np.sqrt(np.diag(cov_mat_s))

        cov_mat_s = cov_mat_s / diag1[:, None] / diag2[None, :]
        cov_mat_s = np.clip(cov_mat_s, -1, 1)
        ds = (math.pi - np.arccos(cov_mat_s)) / math.pi
        cov_mat_s = (cov_mat_s * (math.pi - np.arccos(cov_mat_s)) + np.sqrt(1 - cov_mat_s * cov_mat_s)) / np.pi
        cov_mat_s = cov_mat_s * diag1[:, None] * diag2[None, :]
        return cov_mat_s, ds, diag1

    @staticmethod
    def _adj(cov_matrix_s, adj_block, n1, scale_mat, n2=None):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        n1: number of vertices
        scale_mat: scaling matrix
        """
        if n2 is None:
            n2 = n1
        return adj_block.dot(cov_matrix_s.reshape(-1)).reshape(n1, n2) * scale_mat

    def diag(self, g: Graph):
        """
        compute the diagonal element of GNTK for graph `g`
        g: graph g
        """
        #scale_mat = g.node_scale_coefs() * g.node_scale_coefs()
        num_neigh = [len(neighborhoods) for neighborhoods in g.neighbors()]
        num_neigh = np.array(num_neigh)
        scale_mat = 1/ (num_neigh * num_neigh)
        diag_list = []
        adj_block = sp.kron(g.adj_mat, g.adj_mat)

        # input covariance
        sigma = np.matmul(g.feat_mat, g.feat_mat.T)
        sigma = self._adj(sigma, adj_block=adj_block, n1=g.num_nodes, scale_mat=scale_mat)
        ntk = np.copy(sigma)

        for layer in range(1, self.num_block_ops):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self._next(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_block_ops - 1:
                sigma = self._adj(sigma, adj_block=adj_block, n1=g.num_nodes, scale_mat=scale_mat)
                ntk = self._adj(ntk, adj_block=adj_block, n1=g.num_nodes, scale_mat=scale_mat)
        return diag_list

    def gntk(self, g1: Graph, g2: Graph):
        """
        compute the GNTK value Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """
        diag_list1 = self.diag(g1)
        diag_list2 = self.diag(g2)

        scale_mat = g1.node_scale_coefs() * g2.node_scale_coefs()
        adj_block = sp.kron(g1.adj_mat, g2.adj_mat)

        sigma = np.matmul(g1.feat_mat, g2.feat_mat.T)
        sigma = self._adj(sigma, adj_block=adj_block, n1=g1.num_nodes, n2=g2.num_nodes, scale_mat=scale_mat)
        ntk = np.copy(sigma)

        for layer in range(1, self.num_block_ops):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, _ = self._next(sigma,
                                              diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                              diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_block_ops - 1:
                sigma = self._adj(sigma, adj_block=adj_block, n1=g1.num_nodes, n2=g2.num_nodes, scale_mat=scale_mat)
                ntk = self._adj(ntk, adj_block=adj_block, n1=g1.num_nodes, n2=g2.num_nodes, scale_mat=scale_mat)
        return np.sum(ntk) * 2

    def get_kernel_matrix(self, graphs):

        def calc(pair_ind):
            return self.gntk(graphs[pair_ind[0]], graphs[pair_ind[1]])

        calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

        #pool = Pool(80)
        #results = pool.map(calc, calc_list)
        results = map(calc, calc_list)

        gram = np.zeros((len(graphs), len(graphs)))
        for t, v in zip(calc_list, results):
            gram[t[0], t[1]] = v
            gram[t[1], t[0]] = v

        return gram

class GntkJax:
    def __init__(self, num_mlp_layers: int, num_nodes: int):
        #the width here does not matter
        module = stax.Dense(100)
        layers = []
        for _ in range(num_mlp_layers):
            layers += [module, stax.Relu()]
        layers += [stax.Dense(1)]

        _, _, kernel = stax.serial(*layers)
        self.base_kernel = jit(kernel, static_argnums=(2,))
        self.num_nodes = num_nodes

    def gntk(self, g1: Graph, g2: Graph):
        if g2 is None:
            g2 = g1

        node_feat_aggr_norm1 = g1.feat_mat_aggr_normed()
        node_feat_aggr_norm2 = g2.feat_mat_aggr_normed()

        node_pairs = []
        for feat1 in node_feat_aggr_norm1:
            for feat2 in node_feat_aggr_norm2:
                node_pairs.append((feat1, feat2))

        def calc_pair(pair):
            return self.base_kernel(np.expand_dims(pair[0], axis = 0), np.expand_dims(pair[1], axis = 0), 'ntk')

        ntks = list(map(calc_pair, node_pairs))
        return np.sum(np.array(ntks))/(self.num_nodes**2)

    def get_kernel_matrix(self, graphs1: list, graphs2: Optional[list] = None):
        if graphs2 is None:
            def calc_pair(pair_ind):
                return self.gntk(graphs1[pair_ind[0]], graphs1[pair_ind[1]])

            graph_pairs = [(i, j) for i in range(len(graphs1)) for j in range(i, len(graphs1))]
            results = map(calc_pair, graph_pairs)
            gram = np.zeros((len(graphs1), len(graphs1)))
            for t, v in zip(graph_pairs, results):
                gram[t[0], t[1]] = v
                gram[t[1], t[0]] = v
            return gram
        else:
            def calc_pair(pair_ind):
                return self.gntk(graphs1[pair_ind[0]], graphs2[pair_ind[1]])
            graph_pairs = [(i, j) for i in range(len(graphs1)) for j in range(len(graphs2))]
            results = map(calc_pair, graph_pairs)
            # print(len(results))
            gram = np.zeros((len(graphs1), len(graphs2)))
            for t, v in zip(graph_pairs, results):
                gram[t[0], t[1]] = v
            return gram


if __name__ == '__main__':
    num_nodes = 6
    dim_feats = 5
    sampling_mode = 'binomial'
    edge_prob = 0.1
    data_size = 100

    # g1 = Graph(num_nodes=num_nodes, dim_feats=dim_feats, sampling_mode=sampling_mode, edge_prob=edge_prob)
    # g2 = Graph(num_nodes=num_nodes, dim_feats=dim_feats, sampling_mode=sampling_mode, edge_prob=edge_prob)
    #
    # kernel = GntkDu(num_block_ops=1, num_mlp_layers=3)
    # print(kernel.gntk(g1, g2))
    #
    # rds = np.random.RandomState(800)
    # graph_data = generate_graph_data(data_size=data_size, num_nodes = num_nodes, dim_feats=dim_feats, sampling_mode = sampling_mode, edge_prob=edge_prob, random_state=rds)
    #
    # gram = kernel.get_kernel_matrix(graph_data)
    # print(np.linalg.eigvals(gram))

    g1 = Graph(num_nodes=num_nodes, dim_feats=dim_feats, sampling_mode=sampling_mode, edge_prob=edge_prob)
    g2 = Graph(num_nodes=num_nodes, dim_feats=dim_feats, sampling_mode=sampling_mode, edge_prob=edge_prob)

    kernel = GntkJax(num_nodes= num_nodes, num_mlp_layers=3)
    print(kernel.gntk(g1, g2))
    t = time.time()
    rds = np.random.RandomState(800)
    graph_data = generate_graph_data(data_size=data_size, num_nodes = num_nodes, dim_feats=dim_feats, sampling_mode = sampling_mode, edge_prob=edge_prob, random_state=rds)

    gram = kernel.get_kernel_matrix(graph_data)
    print(np.linalg.eigvals(gram))
    print('Time (mins)', (time.time()-t)/60)
