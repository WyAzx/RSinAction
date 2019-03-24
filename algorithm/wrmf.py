import datetime
import os
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
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
            topn: 推荐结果个数
        """
        self.data = config['data']
        self.test = config['val']
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
        self.topn = config['topn']

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

    def eval_train_tf(self):
        """
        训练模型
        :return:
        """
        tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        self._build_model()
        self.sess = tf.Session(graph=self.input_tensor.graph)
        self.saver = tf.train.Saver([self.row_factor, self.col_factor])

        with self.input_tensor.graph.as_default():
            self.load_tf_model()
            row_update_op = self.model.update_row_factors(sp_input=self.input_tensor)[1]
            col_update_op = self.model.update_col_factors(sp_input=self.input_tensor)[1]

            self.sess.run(self.model.initialize_op)
            self.sess.run(self.model.worker_init)
            for i in range(self.num_iterations):
                self.sess.run(self.model.row_update_prep_gramian_op)
                self.sess.run(self.model.initialize_row_update_op)
                self.sess.run(row_update_op)
                self.sess.run(self.model.col_update_prep_gramian_op)
                self.sess.run(self.model.initialize_col_update_op)
                self.sess.run(col_update_op)
                self.output_row = self.row_factor.eval(session=self.sess)
                self.output_col = self.col_factor.eval(session=self.sess)
                if i % 2 == 0:
                    self.eval_ranking(self.topn)
                    # self.save_tf_model(i)
        tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        self.sess.close()

    def eval_test(self, user_idx):
        """
        获取测试集特定用户的评价物品
        :param user_idx: 用户id
        :return: 测试集用户评价物品列表
        """
        return self.test.getrow(user_idx).indices

    def eval_recommend(self, user_idx, k):
        """
        为特定用户生成推荐列表
        :param user_idx: 用户id
        :param k: 推荐列表大小
        :return: 用户推荐列表
        """
        user_rated = self.data.getrow(user_idx).indices
        assert (self.output_col.shape[0] - len(user_rated)) >= k
        user_f = self.output_row[user_idx]
        pred_ratings = self.output_col.dot(user_f)
        k_r = k + len(user_rated)
        candidate_items = np.argsort(pred_ratings)[-k_r:]
        recommended_items = [i for i in candidate_items if i not in user_rated]
        recommended_items = recommended_items[-k:]
        recommended_items.reverse()

        return recommended_items

    def eval_ranking(self, N):
        """
        对模型进行评价
        :param N: 为每个用户推荐物品的个数
        :return:
        """
        rec_list = {}
        test_list = {}
        for ux in range(len(self.user_map)):
            recommended_items = self.eval_recommend(ux, N)
            rec_list[self.user_map[ux]] = recommended_items
            test_list[self.user_map[ux]] = self.eval_test(ux)
        self.measure = Metrics.rankingMeasure(test_list, rec_list, N)

    def save_tf_model(self, step):
        """
        保存tf模型
        :param step: 全局总步数
        :return:
        """
        self.saver.save(self.sess, os.path.join(self.save_path, 'tf'), global_step=step)

    def load_tf_model(self):
        """
        加载tf模型
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file.")

    def save_model(self):
        """
        使用numpy保存隐矩阵
        :return:
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(os.path.join(self.save_path, 'user'), self.user_map)
        np.save(os.path.join(self.save_path, 'item'), self.item_map)
        np.save(os.path.join(self.save_path, 'row'), self.output_row)
        np.save(os.path.join(self.save_path, 'col'), self.output_col)

    def load_model(self):
        """
        加载隐矩阵
        :return:
        """
        self.user_map = np.load(os.path.join(self.save_path, 'user.npy'))
        self.item_map = np.load(os.path.join(self.save_path, 'item.npy'))
        self.output_row = np.load(os.path.join(self.save_path, 'row.npy'))
        self.output_col = np.load(os.path.join(self.save_path, 'col.npy'))

    def eval_train(self):
        """
        传统方法进行训练
        :return:
        """
        print('Start training...')
        num_rows = self.data.shape[0]
        num_cols = self.data.shape[1]
        if os.path.exists(os.path.join(self.save_path, 'row.npy')) and os.path.exists(os.path.join(self.save_path, 'col.npy')):
            self.load_model()
        else:
            self.output_row = np.random.rand(num_rows, self.dim)
            self.output_col = np.random.rand(num_cols, self.dim)
        iteration = 0
        while iteration < self.num_iterations:
            print ('iteration:',iteration)
            YtY = self.output_col.T.dot(self.output_col)
            I = np.ones(num_cols)
            for uid in range(len(self.user_map)):
                #C_u = np.ones(self.data.getSize(self.recType))
                val = []
                H = np.ones(num_cols)
                pos = []
                P_u = np.zeros(num_cols)
                for iid in self.data.getrow(uid).indices:
                    r_ui = float(self.data.getrow(uid).getcol(iid).toarray()[0][0])
                    pos.append(iid)
                    val.append(r_ui)
                    H[iid]+=r_ui
                    P_u[iid]=1
                    error = (P_u[iid]-self.output_row[uid].dot(self.output_col[iid]))
                C_u = coo_matrix((val,(pos,pos)),shape=(num_cols,num_cols))
                # 计算权重Wu，Wu = (YtCuY + lambda * itemIdx) ^ -1
                Au = (YtY+np.dot(self.output_col.T, C_u.dot(self.output_col))+self.reg * np.eye(self.dim))
                Wu = np.linalg.inv(Au)
                # 更新Xu，这里即X[uid], Xu = Wu*YtCuPu
                self.output_row[uid] = np.dot(Wu,(self.output_col.T*H).dot(P_u))


            XtX = self.output_row.T.dot(self.output_row)
            I = np.ones(num_rows)
            for iid in range(len(self.item_map)):
                P_i = np.zeros(num_rows)
                H = np.ones(num_rows)
                val = []
                pos = []
                for uid in self.data.getcol(iid).indices:
                    r_ui = float(self.data.getrow(uid).getcol(iid).toarray()[0][0])
                    pos.append(uid)
                    val.append(r_ui)
                    H[uid] += r_ui
                    P_i[uid] = 1
                C_i = coo_matrix((val, (pos, pos)),shape=(num_rows,num_rows))
                # 计算权重Wi，Wi = (XtCiX + lambda * userIdx) ^ -1
                Ai = (XtX+np.dot(self.output_row.T,C_i.dot(self.output_row))+self.reg*np.eye(self.dim))
                Wi = np.linalg.inv(Ai)
                # 更新Yi, Yi = Wi*XtCiPi
                self.output_col[iid]=np.dot(Wi, (self.output_row.T*H).dot(P_i))
            iteration += 1
            if iteration % 2 == 0:
                self.save_model()

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
