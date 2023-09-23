import torch
import torch.nn as nn
from utils import *
from MAGNA import MAGNALayer


class BuildSubGraph(nn.Module):
    def __init__(self,
                 adj,
                 num_cates,
                 seq_len,
                 hidden_dim=64,
                 num_layers=2,
                 hop_num=4,
                 alpha=.15,
                 dropout=.5):
        super(BuildSubGraph, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding_cates = nn.Embedding(num_cates, self.hidden_dim)

        self.gdt_layers = nn.ModuleList()
        for l in range(0, self.num_layers):
            self.gdt_layers.append(MAGNALayer(adj, hidden_dim=self.hidden_dim, hop_num=hop_num, 
                                              alpha=alpha, dropout=dropout))        

        self.reset_parameters()
        
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embedding_cates.weight)


    def forward(self, cate_list, mask):
        h = self.embedding_cates.weight
        for l in range(self.num_layers):
            h = self.gdt_layers[l](h)
        graph = h # (n_cates, dim)

        # mask = (cate_list == 0).reshape(cate_list.shape[0], -1) 
        cate_emb = torch.stack([torch.index_select(graph, dim=0, index=cate_list[i]) for i in range(cate_list.shape[0])], 0)
        cate_emb = cate_emb * torch.reshape(mask, (-1, self.seq_len, 1))  # (b, s, d)

        # output = self.interests_model(cate_emb, mask)

        return cate_emb