import torch
import random
from torch.utils.data import DataLoader


class ReadData(object):
    def __init__(self, source):
        self.source = source
        self.n_users, self.n_items, self.n_cates = 0, 0, 0

    def read_data(self):
        self.data_graph = {}
        self.data_len = 0
        self.users = set()
        self.items = set()
        self.cates = set()
        with open(self.source, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(',')
                uid = int(l[0])
                iid = int(l[1])
                cid = int(l[2])
                time = int(float(l[3]))
                self.users.add(uid)
                self.items.add(iid)
                self.cates.add(cid)

                self.n_users = max(self.n_users, uid)
                self.n_items = max(self.n_items, iid)
                self.n_cates = max(self.n_cates, cid)

                if uid not in self.data_graph:
                    self.data_graph[uid] = []
                self.data_graph[uid].append((iid, cid, time))
                self.data_len += 1
        
            for user, value in self.data_graph.items():
                value.sort(key=lambda x: x[2])
                self.data_graph[user] = value

        return self.data_graph, self.data_len 
    
    def max_items_index(self):
        return self.n_users, self.n_items, self.n_cates

    def items_set(self):
        return self.users, self.items, self.cates


class LoadData(torch.utils.data.IterableDataset):
    def __init__(self, source, batch_size=128, seq_len=30, L_time=20, train_flag=0):
        dataset = ReadData(source)
        self.graph, _ = dataset.read_data()
        self.users, self.items, self.cates = dataset.items_set()
        self.users = list(self.users)
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = seq_len
        self.L_time = L_time
        self.index = 0

        print('total users: ', len(list(self.users)))  # 658454
        print('total items: ', len(list(self.items)))  # 675932
        print('total cates: ', len(list(self.cates)))  # 5814

   
    def __iter__(self):
        return self

               
    def __next__(self): 
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size) 
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size
        
        item_id_list = []
        hist_item_list = []
        long_cate_list = []
        short_cate_list = []
        hist_mask_list = []
        short_mask_list = []
        for i, user_id in enumerate(user_id_list):
            user_item_list = self.graph[user_id]
            item_list = [item_[0] for item_ in user_item_list]
            cate_list = [item_[1] for item_ in user_item_list]
            short_cates = [item_[1] - 1 for item_ in user_item_list]
            time_list = [int(item_[2] / (60 * 60 * 24)) for item_ in user_item_list]

            if self.train_flag == 0:
                k = random.choice(range(10, len(user_item_list)))  # 从[10, len(item_list)]中随机选择一个index 
                item_id_list.append(item_list[k]) 
            else:
                k = int(len(user_item_list) * 0.8)
                item_id_list.append(item_list[k:])    

            cates = []
            if k >= self.maxlen:
                hist_item_list.append(item_list[k - self.maxlen: k])               
                long_cate_list.append(cate_list[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
                hist_time_list = time_list[k - self.maxlen: k]
                for n in range(self.maxlen):
                    time_interval = hist_time_list[-1] - hist_time_list[n]
                    if time_interval <= self.L_time:
                        cates.append(short_cates[k - self.maxlen: k][n])
                # short_cate_list.append([0] * (self.maxlen - len(cates)) + cates)   
                # short_mask_list.append([0.0] * (self.maxlen - len(cates)) + [1.0] * len(cates))             
            else:
                hist_item_list.append([0] * (self.maxlen - k) + item_list[:k])  
                long_cate_list.append([0] * (self.maxlen - k) + cate_list[:k]) 
                hist_mask_list.append([0.0] * (self.maxlen - k) + [1.0] * k)     
                hist_time_list = time_list[: k]  
                for n in range(k):
                    time_interval = hist_time_list[-1] - hist_time_list[n]
                    if time_interval <= self.L_time:
                        cates.append(short_cates[:k][n])

            short_cate_list.append([0] * (self.maxlen - len(cates)) + cates) 
            short_mask_list.append([0.0] * (self.maxlen - len(cates)) + [1.0] * len(cates)) 

        return user_id_list, hist_item_list, long_cate_list, short_cate_list, item_id_list, hist_mask_list, short_mask_list


def get_DataLoader(source, batch_size, seq_len, L_time, train_flag):
    dataIterator = LoadData(source, batch_size, seq_len, L_time, train_flag)
    return DataLoader(dataIterator, batch_size=None)


if __name__ == '__main__':
    path = 'data/dataset/'
    train_file = path + 'train.txt'
    train_data = get_DataLoader(train_file, batch_size=128, seq_len=30, L_time= 20, train_flag=0)
    dataiter = iter(train_data)
    (users, items, cates, short_cates, targets, mask)  = dataiter.next()
    # print(items)
    # print(cates)
    # print(targets)
               