# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:19:00 2023

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




def construct_hypergraph_pairwise(data):
    
    adj_pairwise = data.adjacency_matrix().to_dense()
    adj_pairwise_numpy = adj_pairwise.numpy()
    
    pairwise_relations = []
    
    num_nodes = adj_pairwise.shape[0]
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_pairwise_numpy[i, j] == 1:
                L = [i,j]
                pairwise_relations.append(L) 
    
    
    return pairwise_relations
    