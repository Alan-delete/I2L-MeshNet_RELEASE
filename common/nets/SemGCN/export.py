import os, sys
import os.path as osp
import torch
import torch.backends.cudnn as cudnn

sys.path.append(osp.dirname(osp.abspath(__file__)) )
from models.sem_gcn import SemGCN as standard_SemGCN
from common.graph_utils import adj_mx_from_edges

class SemGCN(standard_SemGCN):
    def __init__(self, edges, hid_dim= 128, coords_dim=(2,3), num_layers = 4, nodes_group=None, p_dropout = None):
        num_joints = 0
        for edge in edges:
            num_joints = max (edge[0]+1,edge[1]+1,num_joints )
        adj = adj_mx_from_edges(num_joints,edges, sparse = False)
        #print(adj)
        super().__init__(adj, hid_dim)
#cudnn.benchmark = True

#skeleton= ( (0, 7), (7, 8), (8, 9), (9, 10), (8,11),(11, 12), (12, 13), (8, 14), (14, 15), (15, 16),(0,1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)) 
#sem = SemGCN(skeleton)
