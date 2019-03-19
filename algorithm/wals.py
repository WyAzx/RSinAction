import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import WALSModel

LOG_RATINGS = 0
LINEAR_RATINGS = 1
LINEAR_OBS_W = 100.0

def get_rmse(output_row, output_col, actual):

    mse = 0
    for i in range(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        err = actual.data[i] - np.dot(row_pred, col_pred)
        mse += err * err
    mse /= actual.data.shape[0]
    rmse = math.sqrt(mse)
    return rmse


def simple_train(model, input_tensor, num_iterations):

    sess = tf.Session(graph=input_tensor.graph)

    with input_tensor.graph.as_default():
      row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
      col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

      sess.run(model.initialize_op)
      sess.run(model.worker_init)
      for _ in range(num_iterations):
        sess.run(model.row_update_prep_gramian_op)
        sess.run(model.initialize_row_update_op)
        sess.run(row_update_op)
        sess.run(model.col_update_prep_gramian_op)
        sess.run(model.initialize_col_update_op)
        sess.run(col_update_op)

    return sess

def make_wts(data, wt_type, obs_wt, feature_wt_exp, axis):
    frac = np.array(1.0/(data > 0.0).sum(axis))

    # filter any invalid entries
    frac[np.ma.masked_invalid(frac).mask] = 0.0

    # normalize weights according to assumed distribution of ratings
    if wt_type == LOG_RATINGS:
        wts = np.array(np.power(frac, feature_wt_exp)).flatten()
    else:
        wts = np.array(obs_wt * frac).flatten()

    # check again for any numerically unstable entries
    assert np.isfinite(wts).sum() == wts.shape[0]
    return wts

def wals_model(data, dim, reg, unobs, weights=False,
               wt_type=LINEAR_RATINGS, feature_wt_exp=None,
               obs_wt=LINEAR_OBS_W):
    """
    
    :param data: 
    :param dim: 
    :param reg: 
    :param unobs: 
    :param weights: 
    :param wt_type: 
    :param feature_wt_exp: 
    :param obs_wt: 
    :return: 
    """
    row_wts = None
    col_wts = None

    num_rows = data.shape[0]
    num_cols = data.shape[1]

    if weights:
        assert feature_wt_exp is not None
        row_wts = np.ones(num_rows)
        col_wts = make_wts(data, wt_type, obs_wt, feature_wt_exp, 0)

    with tf.Graph().as_default():
        input_tensor = tf.SparseTensor(indices=zip(data.row, data.col),
                                       values=(data.data).astype(np.float32),
                                       dense_shape=data.shape)

        model = WALSModel(num_rows, num_cols, dim,
                                            unobserved_weight=unobs,
                                            regularization=reg,
                                            row_weights=row_wts,
                                            col_weights=col_wts)

        # retrieve the row and column factors
        row_factor = model.row_factors[0]
        col_factor = model.col_factors[0]

    return input_tensor, row_factor, col_factor, model