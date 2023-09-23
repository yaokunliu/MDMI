import random
import torch
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
                self.data_graph[user] = [(x[0], x[1]) for x in value]

        return self.data_graph, self.data_len 
    
    def max_items_index(self):
        return self.n_users, self.n_items, self.n_cates

    def items_set(self):
        return self.users, self.items, self.cates


class LoadData(torch.utils.data.IterableDataset):
    def __init__(self, source, batch_size=128, seq_len=30, train_flag=0):
        dataset = ReadData(source)
        self.graph, _ = dataset.read_data()
        self.users, self.items, self.cates = dataset.items_set()
        self.users = list(self.users)
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = seq_len
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
        for i, user_id in enumerate(user_id_list):
            user_item_list = self.graph[user_id]
            item_list = [item_[0] for item_ in user_item_list]
            cate_list = [item_[1] for item_ in user_item_list]
            short_cates = [item_[1] - 1 for item_ in user_item_list]

            if self.train_flag == 0:
                k = random.choice(range(10, len(user_item_list)))  
                item_id_list.append(item_list[k]) 
            else:
                k = int(len(user_item_list) * 0.8)
                item_id_list.append(item_list[k:])              
            s = int(self.maxlen * 0.5)  # 10

            if k >= self.maxlen:
                hist_item_list.append(item_list[k - self.maxlen: k])  
                long_cate_list.append(cate_list[k - self.maxlen: k])           
                short_cate_list.append(short_cates[k - s: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                hist_item_list.append(item_list[: k] + [0] * (self.maxlen - k)) 
                long_cate_list.append(cate_list[: k] + [0] * (self.maxlen - k))                 
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))              
                short_len = int(k * 0.5)
                if short_len >= s:
                    short_cate_list.append(short_cates[k - s: k])                                   
                else:
                    short_cate_list.append(short_cates[k - short_len: k] + [0] * (s - short_len))


        return user_id_list, hist_item_list, long_cate_list, short_cate_list, item_id_list, hist_mask_list


def get_DataLoader(source, batch_size, seq_len, train_flag):
    dataIterator = LoadData(source, batch_size, seq_len, train_flag)
    return DataLoader(dataIterator, batch_size=None)


if __name__ == '__main__':
    path = 'data/dataset/'
    train_file = path + 'train.txt'
    train_data = get_DataLoader(train_file, batch_size=128, seq_len=30, train_flag=0)
    # dataiter = iter(train_data)
    # (users, items, cates, short_cates, targets, mask)  = dataiter.next()
    for i, (users, items, long_cates, short_cates, targets, mask) in enumerate(train_data):
        if items == '[]' or long_cates == '[]' or short_cates == '[]' or targets == '[]':
            print(users)
    # print(items)
    # print(cates)
    # print(targets)
               