import datetime
import tensorflow as tf
from .wals import wals_model,simple_train
from utils.data_helper import last_fm_data_loader

class WRMFRecommender(object):

    def __init__(self, config):
        pass

    def build_model(self):
        pass

    def eval_train(self):
        tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        input_tensor, row_factor, col_factor, model = wals_model(train,
                                                                 40,
                                                                 0.1,
                                                                 0,
                                                                 True)

        # factorize matrix
        session = simple_train(model, input_tensor, 10)

        tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        # evaluate output factor matrices
        output_row = row_factor.eval(session=session)
        output_col = col_factor.eval(session=session)

        # close the training session now that we've evaluated the output
        session.close()
        return output_row, output_col

    def eval_test(self):
        pass

    def eval_recommend(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


if __name__ == '__main__':
    users, items, train, test = last_fm_data_loader('dataset/LastFM/user_artists.dat')
    wrmf = WRMFRecommender(None)
    output_row, output_col = wrmf.eval_train(train)
    print(output_row[0])

