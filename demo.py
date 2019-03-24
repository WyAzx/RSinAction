from algorithm.wrmf import WRMFRecommender
from utils.data_helper import data_helper
from config import WRMF_TF_DEFAULT_CONFIG as config

def main():
    users, items, train, val, test = data_helper('dataset/LastFM/user_artists.dat')
    config['data'] = train
    config['val'] = val
    config['user_map'] = users
    config['item_map'] = items
    wrmf = WRMFRecommender(config)
    # wrmf.eval_train() # 传统版本
    wrmf.eval_train_tf() # Tensorflow版本
    wrmf.save_model()
    for ux in val.nonzero()[0]:
        print(wrmf.eval_recommend(ux, config['topn']))

if __name__ == '__main__':
	main()