import os
import sys
import torch
import argparse
from time import time
from utils import *
from model import Model
from evaluation import evaluate
from data_loader import get_DataLoader
from graph_sampler import GraphSampler


torch.autograd.set_detect_anomaly(True)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='test', help='train | test')
    parser.add_argument('--dataset', type=str, default='clothing', help='tmall | taobao | clothing | tafeng')
    parser.add_argument('--path', type=str, default='data/clothing_data/', help='dataset path')
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--hop_num', type=int, default=4)
    parser.add_argument('--group_num', type=int, default=4, help='preferrences/intents num')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN layers')
    parser.add_argument('--alpha', type=float, default=0.2, help='diffusion alpha')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)')  # 最大迭代次数，单位是k（1000）
    parser.add_argument('--patience', type=int, default=70)  # patience，用于early stopping
    parser.add_argument('--gpu', type=str, default='1')  # None -> cpu
    
    args = parser.parse_args()
    return args


def train(device, train_file, valid_file, test_file, graphdata, dataset, batch_size, hidden_dim, seq_len, 
          hop_num, max_iter, test_iter, num_layers, alpha, dp, n_groups, lr, patience, lr_decay):

    exp_name = get_exp_name(dataset, batch_size, lr, hidden_dim, seq_len, n_groups, num_layers, alpha, hop_num, dp)  # 实验名称
    best_model_path = "best_model/" + exp_name + '/'  # 模型保存路径
    # writer = SummaryWriter('runs/' + exp_name)

    print('******loading train data******')
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=0)
    print('******loading valid data******')
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=1)

    model = Model(graphdata, batch_size, hidden_dim, seq_len, n_groups, num_layers, hop_num, alpha, dp)  
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-5
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=lr_decay)

    print('training begin')
    sys.stdout.flush()

    start_time = time()
    trials = 0
    adv_gamma = 0.1
    try:
        total_loss = 0.0
        iters = 0
        best_metric = 0  
        for i, (users, items, long_cates, short_cates, targets, mask) in enumerate(train_data):
            model.train()   
            iters += 1

            optimizer.zero_grad()
            _, loss, grl_loss = model(to_tensor(items, device),
                                      to_tensor(long_cates, device),       
                                      to_tensor(short_cates, device),
                                      to_tensor(targets, device),
                                      to_tensor(mask, device)) 
            
            loss = loss + adv_gamma * grl_loss
            loss.backward()  
            optimizer.step()

            total_loss += loss.item()
        
            if iters % test_iter == 0:
                model.eval()  
                # scheduler.step() 
                
                log_str = 'iter: %d, train loss: %.4f, lr: %.4f, grl_los: %.4f, ' %  (iters, total_loss / test_iter, optimizer.param_groups[0]['lr'], grl_loss)
                
                metrics = evaluate(model, valid_data, hidden_dim, device, 20)
                if metrics != {}:
                    log_str +=  '\n' + ', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()])

                metrics = evaluate(model, valid_data, hidden_dim, device, 50)
                if metrics != {}:
                    log_str +=  ' ' + ', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()])
                
                print(log_str)
                print(exp_name)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:  # early stopping
                            print("early stopping!")
                            break
                    print('trials: ', trials)
                
                # 每次test之后loss_sum置零
                total_loss = 0.0
                test_time = time()
                print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                sys.stdout.flush()
            
            if iters >= max_iter * 1000:  # 超过最大迭代次数，退出训练
                break

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    
    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
    metrics = evaluate(model, valid_data, hidden_dim, device, 20)
    print(', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
    metrics = evaluate(model, valid_data, hidden_dim, device, 50)
    print(', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))

    # 训练结束后用test_data测试一次
    print('******loading test data******')
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=1)
    metrics = evaluate(model, test_data, hidden_dim, device, 20)
    print(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
    metrics = evaluate(model, test_data, hidden_dim, device, 50)
    print(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(device, test_file, graphdata, dataset, batch_size, hidden_dim, 
         seq_len, hop_num, num_layers, alpha, dp, n_groups, lr):
    
    exp_name = get_exp_name(dataset, batch_size, lr, hidden_dim, seq_len, n_groups, num_layers, alpha, hop_num, dp, save=False)  # 实验名称
    best_model_path = "best_model/" + exp_name + '/'  # 模型保存路径A

    model = Model(graphdata, batch_size, hidden_dim, seq_len, n_groups, num_layers, hop_num, alpha, dp)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()
        
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=2)
    metrics = evaluate(model, test_data, hidden_dim, device, 20)
    print(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
    metrics = evaluate(model, test_data, hidden_dim, device, 50)
    print(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))


def main(): 
    pid = os.getpid()
    print('pid:%d' % pid)
    print(sys.argv)
    args = arg_parse()

    if_cuda = torch.cuda.is_available()
    print("if_cuda: ", if_cuda)
    if args.gpu:
        device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        print("use cuda:" + args.gpu if torch.cuda.is_available() else "use cpu, cuda:" + args.gpu + " not available")
    else:
        device = torch.device("cpu")
        print("use cpu")
    
    SEED = args.random_seed
    setup_seed(SEED)


    if args.dataset == 'taobao':
        batch_size = 128
        seq_len = 50
        test_iter = 500
    elif args.dataset == 'tmall':
        batch_size = 128
        seq_len = 50
        test_iter = 500
    elif args.dataset == 'clothing':
        batch_size = 128
        seq_len = 30
        test_iter = 1000
    elif args.dataset == 'retailrocket':
        batch_size = 128
        seq_len = 50
        test_iter = 1000
    elif args.dataset == 'tafeng':
        batch_size = 128
        seq_len = 30
        test_iter = 1000

    train_file = args.path + 'train.txt'
    valid_file = args.path + 'valid.txt'
    test_file = args.path + 'test.txt'
    
    graphdata = GraphSampler(args.path)
    if args.p == 'train':
        train(device, train_file, valid_file, test_file, graphdata, args.dataset, batch_size=batch_size, 
              hidden_dim=args.hidden_size, seq_len=seq_len, hop_num=args.hop_num, max_iter=args.max_iter, 
              test_iter=test_iter, num_layers=args.num_layers, alpha=args.alpha, dp=args.dropout, 
              n_groups=args.group_num, lr=args.learning_rate, patience=args.patience, lr_decay=args.lr_dc)
    elif args.p == 'test':
        test(device, test_file, graphdata, args.dataset, batch_size=batch_size, hidden_dim=args.hidden_size, 
             seq_len=seq_len, hop_num=args.hop_num, num_layers=args.num_layers, alpha=args.alpha, dp=args.dropout, 
             n_groups=args.group_num, lr=args.learning_rate)
    else:
        print('do nothing...')


main()
