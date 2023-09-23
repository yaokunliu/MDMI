import faiss
import math
from utils import *


def evaluate(model, test_data, hidden_size, device, k=20):
    topN = k  # 评价时选取topN

    item_embs = model.output_items().cpu().detach().numpy()  

    res = faiss.StandardGpuResources()  # 使用单个GPU（在单个GPU上运行）
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0  # 使用0号GPU
    
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)  # 建立GPU index用于Inner Product近邻搜索
        gpu_index.add(item_embs)  # 给index添加向量数据
    except Exception as e:
        print("error:", e)
        return

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0

    for _, (users, items, _, short_cates, targets, mask) in enumerate(test_data):  # 一个batch的数据
        user_embs = model(to_tensor(items, device), 
                          None,
                          to_tensor(short_cates, device),
                          None,
                          to_tensor(mask, device),
                          train=False)
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)  # item id
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        else:  # 多兴趣模型评估
            # (batch_size, num_interest, embedding_dim)
            ni = user_embs.shape[1]  # num_interest
            # (batch_size*num_interest, embedding_dim)
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                # print('user: ', users[i])
                # print('item: ', items[i])
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []
                
                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(zip(np.reshape(I[i*ni: (i+1)*ni], -1), np.reshape(D[i*ni: (i+1)*ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序，内积越大，向量越近
                for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                
                # print('labels: ', iid_list)
                # print('recommended items(Top20): ', item_list_set)     
                true_item_set = set(iid_list)  # label去重
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                # print('recall: ', recall)
                # print()
        total += len(targets)  # total增加每个批次的用户数量
        # break

    recall = total_recall / total  # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total  # NDCG
    hitrate = total_hitrate * 1.0 / total  # 命中率
    
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
