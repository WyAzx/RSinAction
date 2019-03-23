import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

def data_helper(filepath='../dataset/LastFM/user_artists.dat'):
    '''
    加载和划分数据集
    :param filepath: 文件路径
    :return: 用户、物品、训练集、测试集
    '''
    users, pds_items, df_items, pv_ratings, ux = data_loader(filepath)
    train, test = data_split(df_items, pv_ratings, ux)

    return users, pds_items.values, train, test


def data_loader(filepath):
    '''
    加载数据集
    :param filepath: 文件路径
    :return: 用户列表、内容ID列表、排序过的内容ID列表、点击量、用户数量
    '''
    df = pd.read_csv(filepath, sep='\t')
    df_items = pd.DataFrame({'contentId': df.artistID.unique()})
    df_sorted_items = df_items.sort_values('contentId').reset_index()
    pds_items = df_sorted_items.contentId
    df_user_items = df.groupby(['userID', 'artistID']).agg({'weight': 'sum'})
    current_u = -1
    ux = -1
    pv_ratings = []
    user_ux = []
    for timeonpg in df_user_items.itertuples():
        user = timeonpg[0][0]
        item = timeonpg[0][1]
        if user != current_u:
            user_ux.append(user)
            ux += 1
            current_u = user
        ix = pds_items.searchsorted(item)
        pv_ratings.append((ux, ix, timeonpg[1]))

    pv_ratings = np.asarray(pv_ratings)
    user_ux = np.asarray(user_ux)

    return user_ux, df_items, pds_items, pv_ratings, ux

def data_split(df_items, pv_ratings, ux):
    '''
    数据集划分
    :param df_items: 内容列表
    :param pv_ratings: 点击量
    :param ux: 用户索引
    :return: 训练集、测试集
    '''
    train, test = train_test_split(pv_ratings, test_size=0.05, random_state=42)
    train = create_sparse(train, ux+1, df_items.size)
    test = create_sparse(test, ux+1, df_items.size)
    return train, test

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