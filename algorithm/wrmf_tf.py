import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import WALSModel


class WRMFRecommender(object):

    def __init__(self, config):
        self.data = config['data']
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

        num_rows = self.data.shape[0]
        num_cols = self.data.shape[1]

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

    def eval_test(self):
        pass

    def eval_recommend(self, user_idx, user_rated, k):
        assert (self.output_col.shape[0] - len(user_rated)) >= k
        user_f = self.output_row[user_idx]
        pred_ratings = self.output_col.dot(user_f)
        k_r = k + len(user_rated)
        candidate_items = np.argsort(pred_ratings)[-k_r:]
        recommended_items = [i for i in candidate_items if i not in user_rated]
        recommended_items = recommended_items[-k:]
        recommended_items.reverse()

        return recommended_items

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
        frac = np.array(1.0 / (data > 0.0).sum(axis))
        frac[np.ma.masked_invalid(frac).mask] = 0.0
        if wt_type == 1:
            wts = np.array(np.power(frac, feature_wt_exp)).flatten()
        else:
            wts = np.array(obs_wt * frac).flatten()
        assert np.isfinite(wts).sum() == wts.shape[0]
        return wts
