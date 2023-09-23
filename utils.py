import os
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def batch_norm(X, X_cate, gamma, beta, moving_mean, moving_var, eps, momentum):
    cate_mean = X_cate.mean(dim=2, keepdim=True)
    cate_std = X_cate.std(dim=2, keepdim=True)
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = cate_std * (X - moving_mean) / torch.sqrt(moving_var + eps) + cate_mean
    else:
        mean = X.mean(dim=1, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=1, keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = cate_std * (X - mean) / torch.sqrt(var + eps) + cate_mean
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    
    Y = gamma * X_hat + beta  # 缩放和移位

    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones((1, 1, num_features)))
        self.beta = nn.Parameter(torch.zeros((1, 1, num_features)))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros((1, 1, num_features))
        self.moving_var = torch.ones((1, 1, num_features))

    def forward(self, X, X_cate):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, X_cate, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        
        return Y


def setup_seed(seed):  # 保证每次运行网络的时候相同输入的输出是固定的
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(seed)  # 为所有的GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # 设置为True，则每次返回的卷积算法是确定的，即默认算法


# 生成实验名称
def get_exp_name(dataset, batch_size, lr, hidden_size, seq_len, group_num, 
                 num_layers, alpha, hop_num, dropout, save=True):
    para_name = '_'.join([dataset, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(hidden_size), 'len' + str(seq_len), 
                          'g' + str(group_num), 'L' + str(num_layers), 'a' + str(alpha), 'h' + str(hop_num), 'dp' + str(dropout), 'code'])
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
