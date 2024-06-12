# Deep Reinforcement Learning for Sequential Combinatorial Auctions

This folder containts the implementation of the paper "Deep Reinforcement Learning for Sequential Combinatorial Auctions"

## Getting Started
The code is written in python3 and requires the following packages
- Numpy
- PyTorch
- Gymnasium
- Stable-baselines3

## Running the Experiments

We consider the following settings:
    
Setting| Setting Name | Description | 
:-------:| :----------------------: | :--------- | 
A | unif    | Additive valuations where item values are independent draws from $U[0, 1]$ |
B | asym    | Additive valuations where item $i$'s values are independent draws over $U[0, \frac{i}{m}]$ |
C | unit    | Unit-demand valuations where item values are independent draws over  $U[0, 1]$ |
D | 3demand | 3-demand valuations where item values are independent draws over  $U[0, 1]$ |   
E | comb1   | Subset valuations are independent draws over $U[0, \sqrt{\|S\|}]$ for every subset        
F | comb2   | Subset valuations are given by $\sum_{j \in T} t_j + c_T$ where $t_j \sim U[1, 2]$ and the complimentarity parameter $c_T \sim U[-\|S\|, \|S\|]$            

We consider the following approaches:

Approach  | Train filename |
:--------:|:--------------|
PPO | train_PPO.py |
DP  (Symmetric) | train_DPsym.py |
DP  (Combinatorial) | train_DPcomb.py |
FPI  | train_FPI.py |

We consider two menu structures:
Menu Structure  | Filename | Notes
:--------:|:--------------| :-------
Combinatorial | bundle.py | Used when num_items <= 10
Entry Fee | entryfee.py | Used when num_items > 10

To run PPO, do
```
python <train_filename> -n <num_agents> -m <num_items> -e <setting_name> -l <learning_rate>
```
To run DP or FPI, do
```
python <train_filename> -n <num_agents> -m <num_items> -e <setting_name>
```

To change other hyperparameters, visit the corresponding file and modify the ```Args``` class.  
The logfiles can be found in ```experiments/``` folder

## Citing the Project

Please cite our work if you find our code/paper is useful to your work.
## Acknowledgements

Parts of the code were adapted from [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3/) and [CleanRL](https://github.com/vwxyzjn/cleanrl) packages
