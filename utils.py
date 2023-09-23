import os
import torch
import random
import shutil
import itertools
import numpy as np
import torch.nn as nn


class SENETLayer(nn.Module):
    def __init__(self, filed_size, reduction_ratio=8):
        super(SENETLayer, self).__init__()
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, filed_size, bias=False),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        Z = torch.mean(inputs, dim=-1, out=None)  # [b, n]
        A = self.excitation(Z)  # [b, n]
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))  # [b, n, d] 

        return V


class BilinearInteraction(nn.Module):
    def __init__(self, filed_size, embedding_size, bilinear_type='each'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()

        if self.bilinear_type == 'all':  # 所有embedding矩阵共用一个矩阵W
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=False)

        elif self.bilinear_type == 'each':
            for _ in range(filed_size):  # 每个field共用一个矩阵W
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))

        elif self.bilinear_type == 'interaction':  # 每个交互用一个矩阵W
            for _, _ in itertools.product(range(filed_size), range(filed_size)):
                self.bilinear.append(nn.Linear(embedding_size, embedding_size, bias=False))

    def forward(self, inputs_A, inputs_B):
        inputs_A = torch.split(inputs_A, 1, dim=1)
        inputs_B = torch.split(inputs_B, 1, dim=1)

        if self.bilinear_type == 'all':  # 所有embedding矩阵共用一个矩阵W
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.product(inputs_A, inputs_B)]

        elif self.bilinear_type == 'each':  # 每个field共用一个矩阵W
            p = [torch.mul(self.bilinear[i](inputs_A[i]), inputs_B[j])
                 for i, j in itertools.product(range(len(inputs_A)), range(len(inputs_B)))]
            # p = [torch.mul(inputs_A[i], inputs_B[j])
            #      for i, j in itertools.product(range(len(inputs_A)), range(len(inputs_B)))]

        elif self.bilinear_type == 'interaction':  # 每个交互用一个矩阵W
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.product(inputs_A, inputs_B), self.bilinear)]

        return torch.cat(p, dim=1)


def setup_seed(seed):  # 保证每次运行网络的时候相同输入的输出是固定的
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed)  # 为所有的GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # 设置为True，则每次返回的卷积算法是确定的，即默认算法


# 生成实验名称
def get_exp_name(dataset, batch_size, lr, hidden_size, seq_len, group_num, num_layers, alpha, L_time, hop_num, dropout, save=True):
    para_name = '_'.join([dataset, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(hidden_size), 'len' + str(seq_len), 
                          'g' + str(group_num), 'L' + str(num_layers), 'a' + str(alpha), 'T' + str(L_time), 'h' + str(hop_num), 'dp' + str(dropout), 'improve'])
    exp_name = para_name

    while os.path.exists('best_model/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('best_model/' + exp_name)  # 递归删除文件夹下的所有子文件夹和子文件
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def to_tensor(var, device):  
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()


def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')


def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'))
    print('model loaded from %s' % path)
