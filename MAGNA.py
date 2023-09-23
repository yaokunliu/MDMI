import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation"""
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


class LayerNorm(nn.Module):
    """Construct a layernorm module"""
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MAGNALayer(nn.Module):
    def __init__(self,
                 adj, 
                 hidden_dim=64,
                 hop_num=4,
                 alpha=.15,
                 dropout=.5):
        super(MAGNALayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.adj = adj

        self.hop_num = hop_num  # 跳数
        self.alpha = alpha  # 扩散参数
        self.feat_drop = nn.Dropout(dropout / 2)  # 0.25
        self.attn_drop = nn.Dropout(dropout)      # 0.5

        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.kaiming_normal_(self.fc_out.weight)
        # entity feed forward
        self.feed_forward = PositionwiseFeedForward(model_dim=self.hidden_dim, d_hidden=4*self.hidden_dim)  
        # entity feed forward normalization
        self.ff_norm = LayerNorm(num_features=self.hidden_dim) 

     
    def forward(self, features):
        rst = self.ppr_estimation(features)
        rst = rst.flatten(1)  # Z (n_cates, dim)
        rst = self.fc_out(rst)  # Z (n_cates, dim)
        rst = features + self.feat_drop(rst)  # ^H_1
        if not self.feed_forward:
            return F.elu(rst)

        rst_ff = self.feed_forward(self.ff_norm(rst))
        rst = rst + self.feat_drop(rst_ff)  # H_1

        return rst  # (n_cates, dim)   

    ######近似注意力扩散的特征聚合######
    def ppr_estimation(self, features):
        feat_0 = self.feat_drop(features)
        feat = feat_0  # Z_0(X)
        # A 边的注意力分数矩阵
        attentions = self.adj  # (n_cates, n_cates)  
        for _ in range(self.hop_num):  # hop_num 跳邻居
            edge_attention = self.attn_drop(attentions)  # 边的权重 (n_cates, n_cates)
            feat = torch.matmul(edge_attention, feat)
            feat = (1.0 - self.alpha) * feat + self.alpha * feat_0
            feat = self.feat_drop(feat) 
        
        return feat  # (n_cates, dim)
