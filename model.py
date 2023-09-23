import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from build_interests import Interests
from build_sub_graphs import BuildSubGraph


class Model(nn.Module):
    def __init__(self,
                 dataset,
                 batch_size=128,
                 hidden_dim=64,  
                 seq_len=30,  
                 n_groups=4,
                 num_layers=2,             
                 hop_num=4,
                 alpha=.15,
                 dropout=.5): 
        super(Model, self).__init__()
        self.num_cates = dataset.n_cates
        self.num_items = dataset.n_items
        self.adj = dataset.get_adj_mat().to_dense()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_groups = n_groups
        self.dropout = nn.Dropout(dropout)
        self.hard_readout = True
        
        self.Embeddings = nn.Embedding(self.num_items + 1, self.hidden_dim, padding_idx=0)
        nn.init.kaiming_normal_(self.Embeddings.weight)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.item_model = Interests(self.num_items, self.num_cates, self.hidden_dim,  
                                    seq_len, self.n_groups, dropout)
        
        self.cate_model = BuildSubGraph(self.adj, self.num_cates, self.hidden_dim, self.n_groups, 
                                        num_layers, hop_num, alpha, dropout)


    def forward(self, item_list, long_cate_list, short_cate_list, label_list, mask, train=True):
        cate_emb = self.cate_model(short_cate_list) 
        item_emb, grl_loss = self.item_model(item_list, long_cate_list, mask, train) 

        if train:
            label_emb = self.Embeddings(label_list)
        
        user_emb = []
        for i in range(self.n_groups):
            key_emb = F.softmax(item_emb * cate_emb[:, i, :].unsqueeze(1), dim=1)
            demand = torch.sum(key_emb * item_emb, dim=1) * cate_emb[:, i, :]
            user_emb.append(self.fc(demand))
        
        user_emb = torch.stack(user_emb, 1)
        
        if not train:
            return user_emb 
        
        readout = self.read_out(user_emb, label_emb)
        scores = self.calculate_score(readout)  
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, label_list)

        return user_emb, loss, grl_loss


    def read_out(self, user_eb, label_eb):
        atten = torch.matmul(user_eb,  # (batch_size, n_groups, dim)  
                             torch.reshape(label_eb, (-1, self.hidden_dim, 1))  # (batch_size, dim, 1)
                             )  # (batch_size, n_groups, 1)  
        # (batch_size, n_groups)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.n_groups)), 1), dim=-1)

        if self.hard_readout:  # 选取n_groups个兴趣胶囊中的一个
            # v_u = V_u[:, argmax(V_u*e_i)]  (batch_size, dim)
            readout = torch.reshape(user_eb, (-1, self.hidden_dim))[  # [b*g, d]
                        # shape=(batch_size, n_groups)
                        (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], 
                                                                    device=user_eb.device) * self.n_groups).long()]
        else:  # 综合n_groups个兴趣胶囊
            #                                           (batch_size, 1, n_groups)
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.n_groups)),
                                   user_eb  # (batch_size, n_groups, dim)
                                   )  # (batch_size, 1, dim)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_dim))  # (batch_size, dim)
        
        # readout是v_u堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）       
        return readout
    

    def calculate_score(self, user_eb):
        all_items = self.Embeddings.weight  # (item_num, dim)
        scores = torch.matmul(user_eb, all_items.transpose(1, 0))  # [b, n]
        return scores
    

    def output_items(self):
        return self.Embeddings.weight