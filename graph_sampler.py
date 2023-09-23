import math
import torch
import numpy as np
from time import time
import scipy.sparse as sp
from data_loader import ReadData


class GraphSampler(object):
    def __init__(self, path='data/dataset/'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        print(f'loading [{path}]')
        self.path = path
        '''
        train_file = path + 'train.txt'
        test_file = path + 'test.txt'
        valid_file = path + 'valid.txt'
        
        self.n_users, self.n_items, self.n_cates = 0, 0, 0

        train_ = ReadData(train_file)
        valid_ = ReadData(valid_file)
        test_ = ReadData(test_file)
        self.train_data, self.n_train = train_.read_data()
        self.valid_data, self.n_valid = valid_.read_data()
        self.test_data, self.n_test = test_.read_data()

        n_users_0, n_items_0, n_cates_0 = train_.max_items_index()
        n_users_1, n_items_1, n_cates_1 = valid_.max_items_index()
        n_users_2, n_items_2, n_cates_2 = test_.max_items_index()
        self.n_users = max(self.n_users, n_users_0, n_users_1, n_users_2)  
        self.n_items = max(self.n_items, n_items_0, n_items_1, n_items_2)  
        self.n_cates = max(self.n_cates, n_cates_0, n_cates_1, n_cates_2)  
        
        print(f"{self.n_users} users" )  # 823068
        print(f"{self.n_items} items" )  # 675979
        print(f"{self.n_cates} categories")  # 5815
        print(f"{self.n_train} interactions for training")  # 58900276
        print(f"{self.n_test} interactions for testing")    # 7376656
        print(f"{self.n_valid} interactions for validing")  # 7353232
        # print(f"Sparsity : {(self.n_train + self.n_test + self.n_valid) / self.n_users / self.n_cates}")  # 0.015384037562358021
        '''
        self.Graph = None
        self.n_cates = 499 # 1297 # 497 # 1442 # 1697 # 
        self.n_items = 293446 # 10176 # 23880 # 311746 # 42286 # 

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def get_adj_mat(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + 's_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_cates, self.n_cates), dtype=np.float32)  # 逐渐添加矩阵中不为0的元素
        
                for u in list(self.train_data.keys()):      
                    seq = self.train_data[u]
                    for i in range(len(seq) - 1): 
                        if adj_mat[seq[i][1] - 1, seq[i + 1][1] - 1] == 0:  # 边的初始值为0
                            adj_mat[seq[i][1] - 1, seq[i + 1][1] - 1] = 1
                        else:
                            adj_mat[seq[i][1] - 1, seq[i + 1][1] - 1] += 1  # 边的出现次数（频率）+ 1       
               
                for i in range(self.n_cates):
                    adj_mat[i, i] = 0  
                
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))  # 度矩阵
                with open('rowsum.txt', 'w') as f:
                    for i in range(len(rowsum)):
                        f.write('%d,%d\n' % (i, rowsum[i]))
                for i in range(self.n_cates):
                    for j in range(self.n_cates):
                        if i != j:
                            adj_mat[i, j] /= (math.sqrt(abs(rowsum[i] + 1) * abs(rowsum[j] + 1)))  # 注意力分数

                norm_adj = adj_mat.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + 's_pre_adj_mat.npz', norm_adj)
                

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)  # 对相同索引的多个值求和, 进行压缩
            print(self.Graph)
            print(self.Graph.shape)  

        return self.Graph


if __name__ == '__main__':
    dataset = GraphSampler()
    adj = dataset.get_adj_mat()
    print(adj.is_sparse)
    adj = adj.to_dense()
    print(adj.is_sparse)  # False
    attentions = torch.nn.functional.softmax(adj, dim=1)
    print(type(attentions))  # <class 'torch.Tensor'>