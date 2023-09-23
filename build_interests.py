import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Interests(nn.Module):
    def __init__(self,
                 num_items, 
                 num_cates,
                 hidden_dim,
                 seq_len,
                 n_groups,
                 dropout,
                 add_pos=True):
        super(Interests, self).__init__()
        self.seq_len = seq_len
        self.num_heads = n_groups
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.feature_classifier = nn.Linear(self.hidden_dim, num_cates + 1)
        self.embedding_items = nn.Embedding(num_items + 1, self.hidden_dim, padding_idx=0)
        
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_dim))  
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4, bias=False),
            nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_dim * 4, self.num_heads, bias=False)
        
        self.reset_parameters()

        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embedding_items.weight)
        nn.init.kaiming_normal_(self.position_embedding)
        nn.init.kaiming_normal_(self.feature_classifier.weight)
        nn.init.constant_(self.feature_classifier.bias, 0)
        
    
    def interests_model(self, item_emb, mask):
        item_emb = torch.reshape(item_emb, (-1, self.seq_len, self.hidden_dim))  # (batch_size, seq_len, dim)
        item_emb = self.dropout(item_emb)

        if self.add_pos:
            item_emb_add_pos = item_emb + self.position_embedding.repeat(item_emb.shape[0], 1, 1)
        else:
            item_emb_add_pos = item_emb

        item_hidden = self.linear1(item_emb_add_pos)  # tanh(W_1*H)  (batch_size, maxlen, dim*4)
        item_att_w = self.linear2(item_hidden)  # W_2*tanh(W_1*H)  (batch_size, maxlen, num_heads)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()  # (batch_size, num_heads, maxlen)

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # (batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # (batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w,  # (batch_size, num_heads, maxlen)
                                    item_emb  # (batch_size, maxlen, dim)
                                    )  # (batch_size, num_heads, dim)

        return interest_emb


    def forward(self, item_list, cate_list, mask, train):
        item_emb = self.embedding_items(item_list)  # (batch_size, seq_len, dim)  
        item_emb = self.dropout(item_emb)      
        item_emb = item_emb * torch.reshape(mask, (-1, self.seq_len, 1))
        
        interests_eb = self.interests_model(item_emb, mask)
        
        if not train:
            return interests_eb, None

        item_feature = item_emb.clone()
        item_feature.register_hook(lambda grad: -grad)
        score_feature = self.feature_classifier(item_feature)  # (b, s, n_cates)
        loss_interaction = self.loss(score_feature, cate_list)

        return interests_eb, loss_interaction


    def loss(self, score, label):
        score = score.view(-1, score.size(-1))
        label = label.view(-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(score, label)   

        return loss