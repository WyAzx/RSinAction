import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

def data_helper(filepath='../dataset/LastFM/user_artists.dat'):
    '''
    构建数据集、划分数据集
    :param filepath: 文件路径
    :return: 用户、物品、训练集、验证集、测试集
    '''
    users, items, data_set = data_loader(filepath)
    train, val, test = data_split(data_set, users.size, items.size)

    return users, items, train, val, test


def data_loader(filepath):
    '''
    加载数据，构建数据集
    :param filepath: 文件路径
    :return: 用户列表、内容列表、数据集
    '''
    df = pd.read_csv(filepath, sep='\t')
    items = pd.DataFrame({'contentId': df.artistID.unique()}).sort_values('contentId').reset_index(drop=True)
    user_items_weight = df.groupby(['userID', 'artistID']).agg({'weight': 'sum'})

    current_u = -1
    ux = -1
    users = []
    data_set = []

    # data_set由[用户index、内容index、权重]构成
    for data in user_items_weight.itertuples():
        user = data[0][0]
        item = data[0][1]
        weight = data[1]

        if user != current_u:
            users.append(user)
            ux += 1
            current_u = user
        ix = items[items['contentId'] == item].index.values[0]
        data_set.append((ux, ix, weight))


    users = np.asarray(users)
    items = np.asarray(items)
    data_set = np.asarray(data_set)

    return users, items, data_set

def data_split(data_set, n_users, n_items):
    '''
    划分数据集
    :param data_set: 数据集
    :param n_users: 用户数量
    :param n_items: 内容数量
    :return:
    '''
    train, test = train_test_split(data_set, test_size=0.05, random_state=42)
    train, val = train_test_split(train, test_size=0.05, random_state=42)
    train = create_sparse(train, n_users, n_items)
    val = create_sparse(val, n_users, n_items)
    test = create_sparse(test, n_users, n_items)
    return train, val, test

def create_sparse(data, n_users, n_items):
    '''
    生成稀疏矩阵
    :param data:数据集
    :param n_users:用户数量
    :param n_items:内容数量
    :return: 稀疏矩阵
    '''
    u_tr, i_tr, r_tr = zip(*data)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))
    return tr_sparse

if __name__ == '__main__':
    users, items, train, test = data_helper()
    print(items)