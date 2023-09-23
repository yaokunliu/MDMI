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
                 dropout=.5, 
                 add_pos=True): 
        super(Model, self).__init__()
        self.num_cates = dataset.n_cates
        self.num_items = dataset.n_items
        self.adj = dataset.get_adj_mat().to_dense()
        self.add_pos = add_pos
        self.seq_len = seq_len
        self.n_groups = n_groups
        self.hard_readout = True
        self.num_heads = n_groups
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size     
        self.dropout = nn.Dropout(dropout) 
        
        self.Embeddings = nn.Embedding(self.num_items + 1, self.hidden_dim, padding_idx=0)

        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_dim)) 
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4, bias=False),
            nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_dim * 4, self.num_heads, bias=False)  
        
        self.item_model = Interests(self.num_items, self.num_cates, self.hidden_dim, seq_len, dropout)
        
        self.cate_model = BuildSubGraph(self.adj, self.num_cates, seq_len, self.hidden_dim, 
                                        num_layers, hop_num, alpha, dropout)
        
        self.reset_parameters()
            

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Embeddings.weight)
        nn.init.kaiming_normal_(self.position_embedding) 


    def forward(self, item_list, long_cate_list, short_cate_list, label_list, mask, short_mask, train=True):
        item_emb = self.cate_model(short_cate_list, short_mask) 
        cate_emb, grl_loss = self.item_model(item_list, long_cate_list, mask, train) 

        if train:
            label_emb = self.Embeddings(label_list)

        user_emb = item_emb + cate_emb
        user_emb = self.interests_model(user_emb, mask)

        if not train:
            return user_emb 
        
        readout = self.read_out(user_emb, label_emb)
        scores = self.calculate_score(readout)  
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, label_list)

        return user_emb, loss, grl_loss


    def interests_model(self, item_emb, mask):
        item_emb = torch.reshape(item_emb, (-1, self.seq_len, self.hidden_dim))  # (b, s, d)
        item_emb = self.dropout(item_emb)

        if self.add_pos:
            item_emb_add_pos = item_emb + self.position_embedding.repeat(item_emb.shape[0], 1, 1)
        else:
            item_emb_add_pos = item_emb

        item_hidden = self.linear1(item_emb_add_pos)  # tanh(W_1*H)  (b, s, d*4)
        item_att_w = self.linear2(item_hidden)  # W_2*tanh(W_1*H)  (b, s, n)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()  # (b, n, s)

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # (b, n, s)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # (b, n, s)

        interest_emb = torch.matmul(item_att_w,  # (b, n, s)
                                    item_emb  # (b, s, d)
                                    )  # (b, n, d)

        return interest_emb
    

    def read_out(self, user_eb, label_eb):
        atten = torch.matmul(user_eb,  # (b, n, d)  
                             torch.reshape(label_eb, (-1, self.hidden_dim, 1))  # (b, d, 1)
                             )  # (b, n, 1)  
        # (b, n)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.n_groups)), 1), dim=-1)

        if self.hard_readout:  # 选取n_groups个兴趣胶囊中的一个
            # v_u = V_u[:, argmax(V_u*e_i)]  (b, d)
            readout = torch.reshape(user_eb, (-1, self.hidden_dim))[  # [b*g, d]
                        # shape=(b, n)
                        (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0], 
                                                                    device=user_eb.device) * self.n_groups).long()]
        else:  # 综合n_groups个兴趣胶囊
            #                                           (b, 1, n)
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.n_groups)),
                                   user_eb  # (b, n, d)
                                   )  # (b, 1, d)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_dim))  # (b, d)
         
        return readout
    

    def calculate_score(self, user_eb):
        all_items = self.Embeddings.weight  # (item_num, dim)
        scores = torch.matmul(user_eb, all_items.transpose(1, 0))  # [b, n]
        return scores
    

    def output_items(self):
        return self.Embeddings.weight
