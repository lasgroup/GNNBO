# Graph Neural Network Bandits
Code to our paper [*Graph Neural Network Bandits*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html), NeurIPS 2022. This paper considers Bandit/Bayesian Optimization problems on finite but very large graph domains, and uses a GNN for construct confidence sets for the target function. 

## Setup

The code is light and self-explanatory. Follows a few pointers to get you started.

* 'requirements.txt' lists the needed packages/used versions. 

* The main two files are `run_phasedgp.py` and `run_gnnucb.py`, which run the GNN-PE and GNN-UCB algorithms, given that the environment is set up.

* This repository does not include the synthetis datasets, but does include the code to generate them from scratch. `graph_env` sub-folder contains the code to the Graph and Reward class, which together make up the environment. Scripts `generate_dataset.py` and `launch_datagen.py`initialize the environment and create the synthetic data. The latter includes a scheduler that parallelizes the process and submits the code to a cluster.

* Alternatively, you can write a data-loader that maps any given dataset into the graph and reward classes defined in the repository. 

* All launcher files are scripts that generate the data for the experiments in our paper and are included for the sake of reproducibility. 

* Implimentation of different BO algorithms are in `algorithms.py` and `net.py` includes our implimentation of a toy GNN. For real-world applications, we suggest using more complex architectures. 
