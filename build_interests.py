import torch
import torch.nn as nn
from utils import *


class Interests(nn.Module):
    def __init__(self,
                 num_items, 
                 num_cates,
                 hidden_dim,
                 seq_len,
                 dropout):
        super(Interests, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout) 

        self.feature_classifier = nn.Linear(self.hidden_dim, num_cates + 1)
        self.embedding_items = nn.Embedding(num_items + 1, self.hidden_dim, padding_idx=0)
        
        self.reset_parameters()

        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embedding_items.weight)
        nn.init.kaiming_normal_(self.feature_classifier.weight)
        nn.init.constant_(self.feature_classifier.bias, 0)       


    def forward(self, item_list, cate_list, mask, train):
        item_emb = self.embedding_items(item_list)  # (b, s, d)     
        item_emb = self.dropout(item_emb)  
        item_emb = item_emb * torch.reshape(mask, (-1, self.seq_len, 1))
        
        # interests_eb = self.interests_model(item_emb, mask)
        
        if not train:
            return item_emb, None

        item_feature = item_emb.clone()
        item_feature.register_hook(lambda grad: -grad)
        score_feature = self.dropout(self.feature_classifier(item_feature))  # (b, s, n)
        loss_interaction = self.loss(score_feature, cate_list)

        return item_emb, loss_interaction


    def loss(self, score, label):
        score = score.view(-1, score.size(-1))
        label = label.view(-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(score, label)   

        return loss
  