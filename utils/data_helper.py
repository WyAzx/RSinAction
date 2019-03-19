import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

def last_fm_data_loader(filepath='../dataset/LastFM/user_artists.dat'):
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
        ix = pds_items.searchsorted(item)[0]
        pv_ratings.append((ux, ix, timeonpg[1]))

    pv_ratings = np.asarray(pv_ratings)
    user_ux = np.asarray(user_ux)

    train, test = train_test_split(pv_ratings, test_size=0.05, random_state=42)
    train = create_sparse(train, ux+1, df_items.size)
    test = create_sparse(test, ux+1, df_items.size)
    return user_ux, pds_items.values, train, test

def create_sparse(data, n_users, n_items):
    u_tr, i_tr, r_tr = zip(*data)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))
    return tr_sparse

if __name__ == '__main__':
    users, items, train, test = last_fm_data_loader()
    print(items)