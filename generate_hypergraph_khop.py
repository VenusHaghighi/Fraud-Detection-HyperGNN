# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:27:40 2023

@author: 46665056
"""

import matplotlib.pyplot as plt
import hypernetx as hnx
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from collections import defaultdict
import random



# convert sparse adjacency matrix to adjacency list

def adjacency_list(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    adj_list = {}

    for node in range(num_nodes):
        neighbors = np.where(adj_matrix[node] == 1)[0]
        adj_list[node] = list(neighbors)

    return adj_list


    
    
    
def construct_hypergraph_with_khop(data, k=2):
    
    #get Khop neighbors 
    adj = dgl.khop_adj(data, 1) + dgl.khop_adj(data, 2) #+ dgl.khop_adj(data, 3)
    adj_numpy = np.array(adj)
    
    #create Identity matrix
    num_nodes = adj.shape[0]
    self_loop_matrix = np.eye(num_nodes)

    final_adj = adj_numpy + self_loop_matrix
    
    #convert adjacency matrix to adjacency list (dict form)
    
    final_adj_dict = adjacency_list(final_adj)
    
    
    ## Remove redundant values from final_adj_dict
    unique_dict = {}
    
    for key, value in final_adj_dict.items():
        # Check if the value is not already in the unique_values_dict
        if value not in unique_dict.values():
            unique_dict[key] = value
    
    # reset the order of keys 
    dict_1 = {i: v for i, v in enumerate(unique_dict.values())}
    
    return dict_1
    
    
    
    
# Initialize a list to store two-hop neighbors
#two_hop_neighbors = set()

# Get one-hop neighbors under the specified relation
#one_hop_neighbors = set(g.successors(target_node, etype=relation_type))

# Loop over one-hop neighbors to find two-hop neighbors
#for neighbor in one_hop_neighbors:
 #   two_hop_neighbors.update(g.successors(neighbor, etype=relation_type))

# Remove the target node itself from the set of two-hop neighbors
#two_hop_neighbors.discard(target_node)


#g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
