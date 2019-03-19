from algorithm.wrmf_tf import WRMFRecommender
from utils.data_helper import last_fm_data_loader

users, items, train, test = last_fm_data_loader('dataset/LastFM/user_artists.dat')
wrmf = WRMFRecommender(None)
output_row, output_col = wrmf.eval_train(train)
print(output_row[0])