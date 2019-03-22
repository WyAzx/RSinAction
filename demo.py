from algorithm.wrmf_tf import WRMFRecommender
from utils.data_helper import last_fm_data_loader
from config import WRMF_TF_DEFAULT_CONFIG as config

users, items, train, test = last_fm_data_loader('dataset/LastFM/user_artists.dat')
config['data'] = train
config['test'] = test
config['user_map'] = users
config['item_map'] = items
wrmf = WRMFRecommender(config)
wrmf.eval_train()
wrmf.eval_ranking(10)
wrmf.save_model()
print(wrmf.eval_recommend(1, [], 10))