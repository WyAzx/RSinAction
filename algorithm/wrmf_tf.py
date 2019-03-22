import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import WALSModel
import sys
sys.path.append("..")
from evaluate.metrics import Metrics

class WRMFRecommender(object):

    def __init__(self, config):
        """
        推荐模型初始化
        :param config:
            data: 训练数据
            user_map: User映射文件
            item_map: Item映射文件
            weight_type: 权重矩阵初始化策略：['user'|'item']
            weights: 是否加权
            wt_type: 权重值线性或指数变换
            obs_wt: 权重线性变换参数
            feature_wt_exp: 权重指数变换参数
            dim: 隐状态维度
            unobs: 缺失值初始化大小
            reg: 正则化参数
            num_iterations: 迭代次数
            save_path: 模型保存路径
        """
        self.data = config['data']
        self.test = config['test']
        self.user_map = config['user_map']
        self.item_map = config['item_map']
        self.weight_type = config['weight_type']
        self.weights = config['weights']
        self.wt_type = config['wt_type']
        self.obs_wt = config['obs_wt']
        self.feature_wt_exp = config['feature_wt_exp']
        self.dim = config['dim']
        self.unobs = config['unobs']
        self.reg = config['reg']
        self.num_iterations = config['num_iterations']
        self.save_path = config['save_path']

        self.output_row = None
        self.output_col = None
        self.row_wts = None
        self.col_wts = None

    def _build_model(self):
        """
        构建wALS算法计算图
        :return:
        """

        num_rows = self.data.shape[0]
        num_cols = self.data.shape[1]

        # Weight矩阵初始化方式
        # 1.User orientation 同一个User下Miss Value平均
        # 2.Item orientation 同一个Item下Miss Value平均
        if self.weights:
            if self.weight_type == 'user':
                self.row_wts = np.ones(num_rows)
                self.col_wts = self._make_wts(self.data, self.wt_type, self.obs_wt, self.feature_wt_exp, 0)
            elif self.weight_type == 'item':
                self.col_wts = np.ones(num_cols)
                self.row_wts = self._make_wts(self.data, self.wt_type, self.obs_wt, self.feature_wt_exp, 1)

        with tf.Graph().as_default():
            self.input_tensor = tf.SparseTensor(indices=list(zip(self.data.row, self.data.col)),
                                                values=(self.data.data).astype(np.float32),
                                                dense_shape=self.data.shape)
            self.model = WALSModel(num_rows, num_cols, self.dim,
                                   unobserved_weight=self.unobs,
                                   regularization=self.reg,
                                   row_weights=self.row_wts,
                                   col_weights=self.col_wts)

            self.row_factor = self.model.row_factors[0]
            self.col_factor = self.model.col_factors[0]

    def eval_train(self):
        """
        训练模型
        :return:
        """
        tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        self._build_model()
        self.sess = tf.Session(graph=self.input_tensor.graph)

        with self.input_tensor.graph.as_default():
            row_update_op = self.model.update_row_factors(sp_input=self.input_tensor)[1]
            col_update_op = self.model.update_col_factors(sp_input=self.input_tensor)[1]

            self.sess.run(self.model.initialize_op)
            self.sess.run(self.model.worker_init)
            for _ in range(self.num_iterations):
                self.sess.run(self.model.row_update_prep_gramian_op)
                self.sess.run(self.model.initialize_row_update_op)
                self.sess.run(row_update_op)
                self.sess.run(self.model.col_update_prep_gramian_op)
                self.sess.run(self.model.initialize_col_update_op)
                self.sess.run(col_update_op)
        tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        self.output_row = self.row_factor.eval(session=self.sess)
        self.output_col = self.col_factor.eval(session=self.sess)
        self.sess.close()

    def eval_test(self, user_idx):
        return self.test.getrow(user_idx).indices

    def eval_recommend(self, user_idx, user_rated, k):
        """
        为特定用户生成推荐列表
        :param user_idx: 用户id
        :param user_rated: 用户已经评价/点击过的Item, 在推荐列表中排除
        :param k: 推荐列表大小
        :return: 用户推荐列表
        """
        assert (self.output_col.shape[0] - len(user_rated)) >= k
        user_f = self.output_row[user_idx]
        pred_ratings = self.output_col.dot(user_f)
        k_r = k + len(user_rated)
        candidate_items = np.argsort(pred_ratings)[-k_r:]
        recommended_items = [i for i in candidate_items if i not in user_rated]
        recommended_items = recommended_items[-k:]
        recommended_items.reverse()

        return recommended_items

    def eval_ranking(self, k):
        rec_list = {}
        test_list = {}
        for ux in range(len(self.user_map)):
            rated_items = self.data.getrow(ux).indices
            recommended_items = self.eval_recommend(ux, rated_items, k)
            rec_list[self.user_map[ux]] = recommended_items
            test_list[self.user_map[ux]] = self.eval_test(ux)
        self.measure = Metrics.rankingMeasure(test_list, rec_list, k)

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(os.path.join(self.save_path, 'user'), self.user_map)
        np.save(os.path.join(self.save_path, 'item'), self.item_map)
        np.save(os.path.join(self.save_path, 'row'), self.output_row)
        np.save(os.path.join(self.save_path, 'col'), self.output_col)

    def load_model(self):
        self.user_map = np.load(os.path.join(self.save_path, 'user'))
        self.item_map = np.load(os.path.join(self.save_path, 'item'))
        self.output_row = np.load(os.path.join(self.save_path, 'row'))
        self.output_col = np.load(os.path.join(self.save_path, 'col'))

    @staticmethod
    def _make_wts(data, wt_type, obs_wt, feature_wt_exp, axis):
        """
        计算缺失值初始化权重
        :param data: 训练数据集
        :param wt_type: 权重线性变换或指数变换
        :param obs_wt: 线性变换参数
        :param feature_wt_exp: 指数变换参数
        :param axis: 数据累加维度
        :return: 在一个维度上权重分布
        """
        frac = np.array(1.0 / (data > 0.0).sum(axis))
        frac[np.ma.masked_invalid(frac).mask] = 0.0
        if wt_type == 1:
            wts = np.array(np.power(frac, feature_wt_exp)).flatten()
        else:
            wts = np.array(obs_wt * frac).flatten()
        assert np.isfinite(wts).sum() == wts.shape[0]
        return wts
