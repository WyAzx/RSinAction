# encoding:utf-8
import math


class Metrics(object):
    """docstring for Metrics"""

    def __init__(self):
        pass

    @staticmethod
    def hits(origin, predict):
        '''
        origin-原始结果
        predict-预测结果
        返回每个用户命中的个数
        '''
        hitCount = {}
        for user in origin.keys():
            items = origin[user]
            predicted = predict[user]
            hitCount[user] = len(set(items) & set(predicted))
        return hitCount

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res.keys():
            if len(origin[user]) == 0:
                continue
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item in origin[user]:
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(res))

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res.keys():
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item in origin[user]:
                    DCG += 1.0/math.log(n+2)
            for n, item in enumerate(origin[user][:N]):
                IDCG += 1.0/math.log(n+2)
            if IDCG == 0:
                continue
            sum_NDCG += DCG / IDCG
        return sum_NDCG / (len(res))

    @staticmethod
    def recall(hits, origin):
        recallList = []
        for user in hits:
            if len(origin[user]) != 0:
                recallList.append(float(hits[user]) / len(origin[user]))
        recall = sum(recallList) / float(len(recallList))
        return recall

    @staticmethod
    def precision(hits, N):
        if len(hits) == 0:
            return 0
        prec = sum([hits[user] for user in hits])
        return float(prec) / (len(hits) * N)

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def rankingMeasure(origin, predicted, N):
        metrics = []
        indicators = []
        # if len(origin) != len(predicted):
        #     print 'The Lengths of test set and predicted set are not match!'
        #     exit(-1)
        hits = Metrics.hits(origin, predicted)
        prec = Metrics.precision(hits, N)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metrics.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        F1 = Metrics.F1(prec, recall)
        indicators.append('F1:' + str(F1) + '\n')
        MAP = Metrics.MAP(origin, predicted, N)
        indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metrics.NDCG(origin, predicted, N)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Metrics.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        metrics.append('Top ' + str(N) + '\n')
        metrics += indicators
        for i in indicators:
            print(i)
        return metrics

    @staticmethod
    def ratingMeasure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:'+str(mae)+'\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse)+'\n')

        return measure

    @staticmethod
    def mae(origin, predict):
        """计算MAE
        Args:
            origin: list, 真实值
            predict: list, 预测值
        Return:
            float MAE
        """
        error = 0
        count = 0
        for ori, pre in zip(origin, predict):
            error += abs(ori - pre)
            count += 1
        if count == 0:
            return error
        return float(error) / count

    @staticmethod
    def rmse(origin, predict):
        """计算RMSE
        Args:
            origin: list, 真实值
            predict: list, 预测值
        Return:
            float RMSE
        """
        error = 0
        count = 0
        for ori, pre in zip(origin, predict):
            error += (ori - pre) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(float(error) / count)
