import numpy as np
from typing import Optional
from itertools import combinations
#from functools import cached_property

class Graph:
    def __init__(self, num_nodes: int, dim_feats: int, sampling_mode: str = 'binomial',
                 adj_mat: Optional[np.ndarray] = None, edge_prob: Optional[float] = None, num_edges: Optional[int] = None,
                 feat_mat: Optional[np.ndarray] = None,
                 random_state=None, **kwargs):
        self.num_nodes = num_nodes
        self.dim_feats = dim_feats
        self._rds = np.random if random_state is None else random_state

        if adj_mat is None:
            if sampling_mode == 'binomial':
                adj_mat = self.sample_binomial_graph(edge_prob)
            elif sampling_mode == 'uniform':
                adj_mat = self.sample_uniform_graph(num_edges)
        self.adj_mat = adj_mat

        if feat_mat is None:
            feat_mat = self.sample_features()
        self.feat_mat = feat_mat


    #@cached_property
    def neighbors(self):
        neighbors = [[] for i in range(self.num_nodes)]
        for node in range(self.num_nodes):
            neighbors[node] = list(np.where(self.adj_mat[node, :] > 0)[0])
        return neighbors

    #@cached_property
    def feat_mat_aggr_normed(self): #matrix of \bar h_u
        normed_feats = np.zeros((self.num_nodes, self.dim_feats))
        neighbors = self.neighbors()
        for node in range(self.num_nodes):
            sum_feats = np.sum(self.feat_mat[neighbors[node]], axis=0)
            normed_feats[node] = sum_feats / np.linalg.norm(sum_feats)
        return normed_feats

    #@cached_property
    def feat_mat_normed(self): # matrix of \bar h_u assuming there are no edges
        norms = np.expand_dims(np.linalg.norm(self.feat_mat, axis=1), axis=1)
        return self.feat_mat / norms

    #@cached_property
    def node_scale_coefs(self):
        # this gives a vector containing the c_u coefs from the GNTK paper. Not used in our implementation
        node_scale = np.zeros(self.num_nodes)
        neighbors = self.neighbors()
        for node in range(self.num_nodes):
            sum_feats = np.sum(self.feat_mat[neighbors[node]], axis=0)
            node_scale[node] = 1/np.linalg.norm(sum_feats)
        return node_scale

    def sample_binomial_graph(self, prob: float):
        adj_mat = self._rds.uniform(0, 1, (self.num_nodes, self.num_nodes))
        adj_mat = adj_mat < prob
        adj_mat = adj_mat.astype(int)
        # this is not symmetric. Flip the lower triangle and put is as the upper
        adj_mat = np.tril(adj_mat) + np.triu(adj_mat.T, 1)
        np.fill_diagonal(adj_mat, 1)
        return adj_mat

    def sample_uniform_graph(self, num_edges: int):
        adj_mat = np.zeros((self.num_nodes, self.num_nodes))
        all_edges = list(combinations(range(self.num_nodes), 2))
        indices = self._rds.choice(range(len(all_edges)), size=num_edges, replace=False)
        picked_edges = [all_edges[i] for i in indices]
        for (i, j) in picked_edges:
            adj_mat[i, j] = 1
            adj_mat[j, i] = 1
        np.fill_diagonal(adj_mat, 1)
        return adj_mat

    def sample_features(self):
        feat_mat = self._rds.normal(size=(self.num_nodes, self.dim_feats))
        return feat_mat

def generate_graph_data(data_size: int, num_nodes: Optional[int], dim_feats: int,
                        sampling_mode: str, edge_prob: Optional[float] = None, num_edges: Optional[int] = None,
                        random_state = None):
    data = []

    # set the random state, needed for reproducability
    rds = np.random if random_state is None else random_state

    # if num_nodes is an int, repeat it
    if type(num_nodes) == int:
        num_nodes = np.repeat(num_nodes, data_size)

    if sampling_mode == 'binomial':
        # if edge_prob is a float, repeat it
        if type(edge_prob) == float:
            edge_prob = np.repeat(edge_prob, data_size)
        for num_node, sampling_param in zip(num_nodes, edge_prob):
            g = Graph(num_nodes=num_node, dim_feats=dim_feats,
                      sampling_mode=sampling_mode, edge_prob=sampling_param,
                      random_state=rds)
            data.append(g)
    elif sampling_mode == 'uniform':
        # if num_edges is an int, repeat it
        if type(num_edges) == int:
            num_edges = np.repeat(num_edges, data_size)
        for num_node, sampling_param in zip(num_nodes, num_edges):
            g = Graph(num_nodes=num_node, dim_feats=dim_feats,
                      sampling_mode=sampling_mode, num_edges=sampling_param,
                      random_state=rds)
            data.append(g)
    else:
        raise NotImplementedError

    return data


if __name__ == '__main__':
    num_nodes = 5
    dim_feats = 2
    sampling_mode = 'binomial'
    prob = 0.5
    seed = 800
    g = Graph(num_nodes=num_nodes, dim_feats=dim_feats, sampling_mode=sampling_mode, edge_prob=prob)
    rds = np.random.RandomState(seed)
    graph_data = generate_graph_data(data_size=10, num_nodes = 5, dim_feats=10, sampling_mode = 'binomial', edge_prob=0.5, random_state=rds)
