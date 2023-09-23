import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from MAGNA import MAGNALayer
from time import time


class BuildSubGraph(nn.Module):
    def __init__(self,
                 adj,
                 num_cates,
                 hidden_dim=64,
                 n_groups=4,
                 num_layers=2,
                 hop_num=4,
                 alpha=.15,
                 dropout=.5):
        super(BuildSubGraph, self).__init__()
        self.n_groups = n_groups
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        self.embedding_cates = nn.Embedding(num_cates, self.hidden_dim)

        self.gdt_layers = nn.ModuleList()
        for l in range(0, self.num_layers):
            self.gdt_layers.append(MAGNALayer(adj, hidden_dim=self.hidden_dim, hop_num=hop_num, 
                                              alpha=alpha, dropout=dropout))
        
        ###### subgraph_cluster parameters ######
        self.bn = BatchNorm(num_features=self.hidden_dim)
        self.group_func = nn.Linear(self.hidden_dim, self.n_groups)  # U_(k)
        self.group_pool_func = nn.Linear(num_cates, 1)  # U'_(k)
        
        self.reset_parameters()
        
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embedding_cates.weight)
        nn.init.kaiming_normal_(self.group_func.weight)
        nn.init.kaiming_normal_(self.group_pool_func.weight)
        nn.init.constant_(self.group_func.bias, 0)
        nn.init.constant_(self.group_pool_func.bias, 0)
     

    def forward(self, cate_list):
        h = self.embedding_cates.weight
        for l in range(self.num_layers):
            h = self.gdt_layers[l](h)
        graph = h # (n_cates, dim)

        # 分配矩阵 S (n_cates, n_groups)
        score_cluster = F.softmax(self.group_func(graph), dim=1)  
        # (n_groups, n_cates, dim)
        sub_graph = torch.stack([score_cluster[:, group].unsqueeze(dim=1) * graph for group in range(self.n_groups)], dim=0)
        mask = (cate_list == 0).reshape(-1, 1, cate_list.shape[1], 1) 

        seq_emb = torch.stack([torch.index_select(sub_graph, dim=1, index=cate_list[i]) for i in range(cate_list.shape[0])], 0)
        seq_emb = seq_emb.masked_fill(mask, 0)
        # (batch_size, n_groups, n_cates, dim)
        user_sub_graph = self.bn(sub_graph, seq_emb)
    
        # 池化
        user_sub_graph = torch.transpose(user_sub_graph, 3, 2).contiguous() 
        # output = self.dropout(F.softmax(self.group_pool_func(user_sub_graph), dim=1))
        output = self.dropout(self.group_pool_func(user_sub_graph))
        # output = self.dropout(torch.mean(user_sub_graph, dim=2))  # (batch_size, n_groups, dim)
        output = output.squeeze()

        return output
