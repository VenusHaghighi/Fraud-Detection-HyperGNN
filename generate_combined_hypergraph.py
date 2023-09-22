# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:25:13 2023

@author: 46665056
"""

from prepare_data import *
from generate_hypergraph_khop import *
from generate_hypergraph_Eudis import *
from generate_hypergraph_pairwise import *


class Dataset:
    def __init__(self, name='yelp'):
        self.name = name
        graph = None
        
        if name == 'yelp':
            
                dataset = FraudYelpDataset()
                graph = dataset[0]

        self.graph = graph




if __name__ == '__main__':
    
    graph = Dataset('yelp').graph
    print(graph)
    feature = graph.ndata['feature']
    graph = graph.edge_type_subgraph(['net_rur'])
    print(graph)
    
    
    # creating hyperedges dictionary based on khop neighbors
    dict_1 = construct_hypergraph_with_khop(graph, k=2)
    
    # creating hyperedges dictionary based on KNN
    dict_2 = construct_hypergraph_with_KNN(feature, k=3)
    
    # creating hyperedges dictionary based on pairwise relations
    pairwise_list = construct_hypergraph_pairwise(graph)
    
    
    
    # combining different groups of hyperedges to generate the final dictionary of hyperedges
    list_values = list(dict_1.values()) + list(dict_2.values()) + pairwise_list
    
    
    print("number of hyper edges in dict_1:", len(list(dict_1.values())))
    print("number of hyper edges in dict_2:", len(list(dict_2.values())))
    print("number of hyper edges in dict_3:", len(pairwise_list))
    
    
    set_values = set(tuple(item) for item in list_values)
    filtered_set = {lst for lst in set_values if len(lst) > 1}
    list_values_unique = list(set_values)
    
    final_dict = {index: value for index, value in enumerate(list_values_unique)}
    
    print(final_dict)
    print("number of hyper edges in final_dict:", len(list(final_dict.values())))
    
    torch.save(final_dict, "final_dict.pt")