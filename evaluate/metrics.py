# encoding:utf-8
import math


class Metrics(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, predict):
        '''
        计算命中个数
        :param origin: 原始结果
        :param predict: 预测结果
        :return: 用户命中的个数字典
        '''
        hitCount = {}
        for user in origin.keys():
            items = origin[user]
            predicted = predict[user]
            hitCount[user] = len(set(items) & set(predicted))
        return hitCount

    @staticmethod
    def MAP(origin, predict, N):
        '''
        计算MAP(Mean Average Precision)
        :param origin: 原始结果
        :param predict: 预测结果
        :param N: 为每个用户推荐物品的个数
        :return: MAP
        '''
        sum_prec = 0
        for user in predict.keys():
            if len(origin[user]) == 0:
                continue
            hits = 0
            precision = 0
            for n, item in enumerate(predict[user]):
                if item in origin[user]:
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(predict))

    @staticmethod
    def NDCG(origin, predict, N):
        '''
        计算NDCG(Normalized Discounted Cumulative Gain)
        :param origin: 原始结果
        :param predict: 预测结果
        :param N: 为每个用户推荐物品的个数
        :return: NDCG
        '''
        sum_NDCG = 0
        for user in predict.keys():
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(predict[user]):
                if item in origin[user]:
                    DCG += 1.0/math.log(n+2)
            for n, item in enumerate(origin[user][:N]):
                IDCG += 1.0/math.log(n+2)
            if IDCG == 0:
                continue
            sum_NDCG += DCG / IDCG
        return sum_NDCG / (len(predict))

    @staticmethod
    def recall(hits, origin):
        '''
        计算召回率(Recall)
        :param hits: 用户命中个数
        :param origin: 原始结果
        :return: 召回率
        '''
        recallList = []
        for user in hits:
            if len(origin[user]) != 0:
                recallList.append(float(hits[user]) / len(origin[user]))
        recall = sum(recallList) / float(len(recallList))
        return recall

    @staticmethod
    def precision(hits, N):
        '''
        计算精确率(Precision)
        :param hits: 用户命中个数
        :param N: 为每个用户推荐物品的个数
        :return: 精确率
        '''
        if len(hits) == 0:
            return 0
        prec = sum([hits[user] for user in hits])
        return float(prec) / (len(hits) * N)

    @staticmethod
    def F1(prec, recall):
        '''
        计算F1值
        :param prec: 精确率
        :param recall: 召回率
        :return: F1值
        '''
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def ranking_measure(origin, predict, N):
        '''
        计算TopN类型推荐算法评价指标
        :param origin: 原始结果
        :param predict: 预测结果
        :param N: 为每个用户推荐物品的个数
        :return: 包含评价指标的列表
        '''
        print('Top', N)
        measure = []
        hits = Metrics.hits(origin, predict)
        prec = Metrics.precision(hits, N)
        measure.append(prec)
        recall = Metrics.recall(hits, origin)
        measure.append(recall)
        F1 = Metrics.F1(prec, recall)
        measure.append(F1)
        MAP = Metrics.MAP(origin, predict, N)
        measure.append(MAP)
        NDCG = Metrics.NDCG(origin, predict, N)
        measure.append(NDCG)
        print('Precision:', prec)
        print('Recall:', recall)
        print('F1:', F1)
        print('MAP:', MAP)
        print('NDCG:', NDCG)
        return measure