# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:10:23 2023

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



def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def construct_hyperedges_with_KNN_from_distance(dis_mat, k):
    """
    construct hypregraph dictionary from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k: K nearest neighbor parameter
    :return: a dictionary which includes hyperedges
    """
    num_nodes= dis_mat.shape[0]
    dic ={}
    for idx in range(num_nodes):
        
        dis_vec = dis_mat[idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        
        # select K nearest neighbors for each individual node based on the pair-wise calculated Euclidian distance
        nearest_idx = nearest_idx.tolist()[0:k+1]
        # create hyperedges dictionary
        dic[idx]=nearest_idx
    print(dic)    
    return dic


def construct_hypergraph_with_KNN(X, k):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    
    dis_mat = Eu_dis(X)
    
    dict_2 = construct_hyperedges_with_KNN_from_distance(dis_mat, k)

    return dict_2




# saving and loading the hypergraph as json file   
 
#with open("C:/Users/46665056/Documents/HyperG.json", 'w') as file:
        #json.dump(dic, file)
        
        
#with open("C:/Users/46665056/Documents/HyperG.json", 'r') as file:
    #load_data = json.load(file)