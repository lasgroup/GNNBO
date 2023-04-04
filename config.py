import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

alg_lambdas = {
    'GNN_UCB': [0.006292296014580299, 0.00019231738847013348],
    'NN_UCB': [2.50271959e-03, 5.2728189412304615e-05],
    'GNN_US': [1.4381179896541183, 3.881227769982059e-06],
    'NN_US': [6.725439193699782, 0.08383616432052483]
}
alg_betas = {
    'GNN_UCB': [0.0012478313525392873, 0.8663146356523006],
    'NN_UCB': [2.34528090e-04, 0.0007216093834916769],
    'GNN_US': [7.745362422474653e-06, 0.0790891675045115],
    'NN_US': [1.1511950867110118e-06, 0.00013966344084855373],
}

alg_pretrain_steps = {
    'GNN_UCB':[40,40],
    'NN_UCB': [40,40],
    'GNN_US': [43,48],
    'NN_US': [63,63]
}
alg_intersect = {
    'GNN_US': [80,48],
    'NN_US': [80,63]
}