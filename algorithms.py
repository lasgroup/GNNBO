import numpy as np
import torch.optim as optim
import copy
from nets import NN, GNN, normalize_init
from typing import Optional
from config import *

class UCBalg:
    def __init__(self, net: str, feat_dim: int, num_nodes: int, num_mlp_layers: int = 1, alg_lambda: float = 1,
                 exploration_coef: float = 1, neuron_per_layer: int = 100,
                 complete_cov_mat: bool = False, lr: float = 1e-3,
                 random_state = None, nn_aggr_feat: bool = True,
                 train_from_scratch=False, verbose = True,
                 path: Optional[str] = None, **kwargs):

        if net == 'NN':
            self.func = NN(input_dim=feat_dim * num_nodes, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(device)
        elif net == 'GNN':
            self.func = GNN(input_dim=feat_dim, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(device)
        else:
            raise NotImplementedError

        self._rds = np.random if random_state is None else random_state
        self.alg_lambda = alg_lambda  # lambda regularization for the algorithm
        self.num_net_params = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = alg_lambda * torch.ones((self.num_net_params,)).to(device)
        self.U_inv_small = None
        self.exploration_coef = exploration_coef
        self.neuron_per_layer = neuron_per_layer
        self.train_from_scratch = train_from_scratch

        self.complete_cov_mat = complete_cov_mat  # if true, the complete covariance matrix is considered. Otherwise, just the diagonal.
        self.lr = lr

        self.verbose = verbose

        if path is None:
            self.path = 'trained_models/{}_{}dim_{}L_{}m_{:.3e}beta_{:.1e}sigma'.format(net, feat_dim, num_mlp_layers,
                                                                                        neuron_per_layer,
                                                                                        self.exploration_coef,
                                                                                        self.alg_lambda)
        else:
            self.path = path

        self.G = None

    def get_infogain(self):
        kernel_matrix = self.U - self.alg_lambda * torch.ones((self.num_net_params,)).to(device)
        return 0.5 * np.log(torch.prod(1 + kernel_matrix / self.alg_lambda).cpu().numpy())

    def save_model(self):
        torch.save(self.func, self.path)

    def load_model(self):
        try:
            self.func = torch.load(self.path)
            self.func.eval()
        except:
            print('Pretrained model not found.')


class GnnUCB(UCBalg):  # Our main method
    # This class currently uses Woodbury's Identity. For scalability experiment, we need to use the regular gradient.
    def __init__(self, net: str,num_nodes: int, feat_dim: int, num_actions: int, action_domain: list,
                 alg_lambda: float = 1, exploration_coef: float = 1, t_intersect: int = np.inf,
                 num_mlp_layers: int = 2, neuron_per_layer: int = 128, lr: float = 1e-3,
                 nn_aggr_feat = True, train_from_scratch = False, verbose = True,
                 nn_init_lazy: bool = True, complete_cov_mat: bool = False, random_state = None, path: Optional[str] = None, **kwargs):
        super().__init__(net=net, feat_dim=feat_dim, num_mlp_layers=num_mlp_layers, alg_lambda=alg_lambda, verbose = verbose,
                         lr = lr, complete_cov_mat = complete_cov_mat, nn_aggr_feat = nn_aggr_feat, train_from_scratch = train_from_scratch, num_nodes=num_nodes,
                         exploration_coef=exploration_coef, neuron_per_layer=neuron_per_layer, random_state=random_state, path=path, **kwargs)


        # Create the network for computing gradients and subsequently variance.
        self.f0 = copy.deepcopy(self.func)
        self.f0 = normalize_init(self.f0)

        if nn_init_lazy:
            self.func = normalize_init(self.func)
            self.f0 = copy.deepcopy(self.func)

        if net == 'NN':
            self.name = 'NN-UCB'
        else:
            self.name = 'GNN-UCB'

        self.data = {
            'graph_indices': [],
            'rewards': []
        }

        self.num_actions = num_actions
        self.action_domain = action_domain

        self.init_grad_list = []
        self.get_init_grads()

    def save_model(self):
        super().save_model()
        torch.save(self.f0, self.path + "/f0_model")

    def get_init_grads(self):
        post_mean0 = []
        for graph in self.action_domain:
            self.f0.zero_grad()
            post_mean0.append(self.f0(graph))
            post_mean0[-1].backward(retain_graph=True)
            # Get the Variance.
            g = torch.cat([p.grad.flatten().detach() for p in self.f0.parameters()])
            self.init_grad_list.append(g)

    # def get_small_cov(self, g: np.ndarray):
    #     # Need to check square root. In any case, it is an issue of scaling - sweeping over beta properly would work.
    #     k_xx = g.dot(g)
    #     k_xy = torch.matmul(g.reshape(1, -1), self.G.T)
    #     k_xy = torch.matmul(k_xy, self.U_inv_small)
    #     k_xy = torch.matmul(k_xy, torch.matmul(self.G, g.reshape(-1, 1)))
    #     final_val = k_xx - k_xy
    #     return final_val

    def add_data(self, indices, rewards):
        # add the new observations, only if it didn't exist already
        for idx, reward in zip(indices, rewards):
            #if idx not in self.data['graph_indices']: #TODO: uncomment?
            self.data['graph_indices'].append(idx)
            self.data['rewards'].append(reward)

            #g_to_add = self.init_grad_list[idx]
            if self.complete_cov_mat:
                raise NotImplementedError
                # if self.G is None:
                #     self.G = g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)
                # else:
                #     self.G = torch.cat((self.G, g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)), dim=0)
                #
                # kernel_matrix = torch.matmul(self.G, self.G.t())
                # self.U_inv_small = torch.inverse(
                #     torch.diag(torch.ones(self.G.shape[0]).to(device) * self.alg_lambda) + kernel_matrix)
            else:
                self.U += self.init_grad_list[idx] * self.init_grad_list[idx] / self.neuron_per_layer  # U is diagonal

    def select(self):
        ucbs = []
        for ix in range(self.num_actions):
            post_mean = self.func(self.action_domain[ix])
            g = self.init_grad_list[ix]
            if self.complete_cov_mat:
                raise NotImplementedError
                # if self.G is None:
                #     post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g / self.U / self.neuron_per_layer))
                # else:
                #     post_var = np.sqrt(self.exploration_coef) * torch.sqrt(
                #         self.get_small_cov(g / np.sqrt(self.neuron_per_layer)))
            else:
                # Use Approximate Covariance.
                post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer)
            ucbs.append(post_mean.item() + np.sqrt(self.exploration_coef) * post_var.item())
        ix = np.argmax(ucbs)
        return ix

    def explore(self):
        ix = self._rds.choice(range(self.num_actions))
        return ix

    def exploit(self):
        if len(self.data['rewards'])>0:
            list_ix = np.argmax(self.data['rewards'])
            ix = self.data['graph_indices'][list_ix]
            return ix
        else:
            return self.explore()

    def best_predicted(self):
        means = []
        for ix in range(self.num_actions):
            post_mean = self.func(self.action_domain[ix])
            means.append(post_mean.item())
        ix = np.argmax(means)
        return ix

    def get_post_var(self, idx):
        g = self.init_grad_list[idx]
        if self.complete_cov_mat:
            raise NotImplementedError
        else:
            return torch.sqrt(torch.sum(g * g / self.U ) / self.neuron_per_layer).item()

    def get_post_mean(self, idx):
        return self.func(self.action_domain[idx]).item()

    def pretrain(self, pre_train_data):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=self.alg_lambda/100)
        self.data['graph_indices'].extend(pre_train_data['graph_indices'])
        self.data['rewards'].extend(pre_train_data['rewards'])
        index = list(np.arange(len(self.data['graph_indices'])))
        length = len(index)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            epoch_loss = 0
            for ix in index:
                label = self.data['rewards'][ix]
                optimizer.zero_grad()
                delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                loss = delta * delta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 100:  # train each epoch for J \leq 1000
                    return tot_loss / 1000
            if epoch_loss / length <= 1e-6:  # stop training if the average loss is less than 0.001
                return epoch_loss / length

    def train(self):
        if self.train_from_scratch:
            self.func.load_state_dict(self.f0.state_dict())

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)#, weight_decay=self.alg_lambda)
        #optimizer = optim.SGD(self.func.parameters(), lr=self.lr, weight_decay=self.alg_lambda)

        # index = list(np.arange(len(self.data['graph_indices'])))
        # # cnt = 0
        # # tot_loss = 0
        # for __ in range(10):
        #     epoch_loss = 0
        #     self._rds.shuffle(index)
        #     for ix in index:
        #         label = self.data['rewards'][ix]
        #         optimizer.zero_grad()
        #         delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
        #         loss = delta * delta
        #         loss.backward()
        #         optimizer.step()
        #         epoch_loss += loss.item()
        #         # tot_loss += loss.item()
        #         # cnt += 1
        # # return tot_loss / len(index) / 10
        # #self.save_model()

        index = list(np.arange(len(self.data['graph_indices'])))
        length = len(index)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            for ix in index:
                label = self.data['rewards'][ix]
                optimizer.zero_grad()
                delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                loss = delta * delta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                    return tot_loss / cnt
            delta2 = epoch_losses[-2]-epoch_losses[-1]
            delta1 = epoch_losses[-3]-epoch_losses[-2]
            relative_improvement = (delta1-delta2)/delta1
            if relative_improvement < 0.001:
                if self.verbose:
                    print('Loss curve is getting flat, and the count is', cnt)
                    print('The loss is', epoch_loss / length)
                return epoch_loss / length
            if epoch_loss / length <= 1e-4:  # stop training if the average loss is less than 0.0001
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length


class PhasedGnnUCB(GnnUCB):
    def __init__(self, net: str,num_nodes: int, feat_dim: int, num_actions: int, action_domain: list,
                 alg_lambda: float = 1, exploration_coef: float = 1, t_intersect: int= np.inf,
                 num_mlp_layers: int = 2, neuron_per_layer: int = 128, lr: float = 1e-3,
                 nn_aggr_feat = True, train_from_scratch = False, verbose = True,
                 nn_init_lazy: bool = True, complete_cov_mat: bool = False, random_state = None, path: Optional[str] = None, **kwargs):
        super().__init__(net = net, num_nodes = num_nodes, feat_dim = feat_dim, num_actions = num_actions, action_domain = action_domain,
        alg_lambda = alg_lambda, exploration_coef = exploration_coef,
        num_mlp_layers = num_mlp_layers, neuron_per_layer = neuron_per_layer, lr = lr,
        nn_aggr_feat = nn_aggr_feat, train_from_scratch = train_from_scratch, verbose = verbose,
        nn_init_lazy = nn_init_lazy, complete_cov_mat = complete_cov_mat, random_state = random_state, path = path)
        self.maximizers = [i for i in range(self.num_actions)]
        self.t_intersect = t_intersect
        if net == 'NN':
            self.name = 'PhasedNN-UCB'
        else:
            self.name = 'PhasedGNN-UCB'

    def select(self):
        ucbs = []
        lcbs = []
        vars = []
        for ix in range(self.num_actions):
            post_mean = self.func(self.action_domain[ix])
            g = self.init_grad_list[ix]
            post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer)
            vars.append(post_var.item())
            ucbs.append(post_mean.item() + np.sqrt(self.exploration_coef ) * post_var.item())
            lcbs.append(post_mean.item() - np.sqrt(self.exploration_coef ) * post_var.item())
            # ucbs.append(post_mean.item() + np.sqrt(self.exploration_coef * ) * post_var.item())
            # lcbs.append(post_mean.item() - np.sqrt(self.exploration_coef * ) * post_var.item())
        #max_lcb = np.max(lcbs)
        t = len(self.data['graph_indices'])
        if t > self.t_intersect:
            max_lcb = np.max([lcbs[i] for i in self.maximizers])
            self.maximizers = [i for i in self.maximizers if max_lcb <= ucbs[i]]
            print('intersecting...')
        else:
            max_lcb = np.max(lcbs)
            self.maximizers = [i for i in range(len(ucbs)) if max_lcb <= ucbs[i]]
        maximizer_vars = [vars[i] for i in self.maximizers]
        ix = self.maximizers[np.argmax(maximizer_vars)]
        return ix

    def train(self):
        if self.train_from_scratch:
            self.func.load_state_dict(self.f0.state_dict())

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)#, weight_decay=self.alg_lambda)

        index = list(np.arange(len(self.data['graph_indices'])))
        length = len(index)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            for ix in index:
                label = self.data['rewards'][ix]
                optimizer.zero_grad()
                delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                loss = delta * delta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                    return tot_loss / cnt
            delta2 = epoch_losses[-2]-epoch_losses[-1]
            delta1 = epoch_losses[-3]-epoch_losses[-2]
            relative_improvement = (delta1-delta2)/delta1
            if relative_improvement < 0.001:
                if self.verbose:
                    print('Loss curve is getting flat, and the count is', cnt)
                    print('The loss is', epoch_loss / length)
                return epoch_loss / length
            if epoch_loss / length <= 1e-4:  # stop training if the average loss is less than 0.0001
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length

