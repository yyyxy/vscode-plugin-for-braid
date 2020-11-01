import split_data, feedback, metric
import xgboost as xgb
import gensim
import _pickle as pickle
import csv
import numpy as np


#first LTR only
def get_LTR_feature(t_answer, t_rec_api, feature):
    training_feature = []
    for i, train in enumerate(t_answer):
        # print('train',i, train)
        for index, ap in enumerate(t_rec_api[i*10:i*10+10]):
            if ap in train:
                temp = [1]
            else:
                temp = [0]
            temp.extend(feature[i*10+index][1:])
            training_feature.append(temp)
            # print(temp)
    return training_feature


def get_LTR_predict(test_feature, train_x_feature, train_y_feature):
    # X_train, y_train = split_data.get_train_feature_matrix(train_feedback_info, choose_feature)
    # X_test = split_data.get_test_feature_matrix(test_feedback_info, test_feature)
    X_test = test_feature

    dtrain = xgb.DMatrix(train_x_feature, train_y_feature)
    num_rounds = 100
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_test)

    y_predict = model.predict(dtest)
    y_predict = y_predict.tolist()

    return y_predict


def get_LTR_predict_LTR(test_feature, test_feedback_info, choose_feature, train_feedback_info):
    X_train, y_train = split_data.get_train_feature_matrix(train_feedback_info, choose_feature)
    X_test = split_data.get_test_feature_matrix(test_feedback_info, test_feature)

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 100
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_test)

    y_predict = model.predict(dtest)
    y_predict = y_predict.tolist()
    return y_predict


params = {
        'booster': 'gbtree',
        'objective': 'rank:pairwise',
        'gamma': 0.3,
        'max_depth': 3,
        'lambda': 1,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.01,
        'seed': 100,
        'alpha': 1,
        'nthread': -1,
        'eval_metric': 'map@1-',
    }


if __name__ == "__main__":
    w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')  # pre-trained word embedding
    idf = pickle.load(open('../data/idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary

    test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train = split_data.get_test_train()
    print('test_answer', test_answer)
    LTR_train_feature = get_LTR_feature(train_answer, rec_api_train, train_feature)
    num_choose = 373
    top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
    # iteration begin
    for round in range(1):
        choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_choose_unlabel(
            train_query, train_answer, rec_api_train, num_choose)
        choose_feature, unlabel_feature = split_data.get_choose(LTR_train_feature, choose)

        train_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
        test_feedback_info = feedback.get_feedback_inf(test_query, choose_query, choose_answer, rec_api_test, w2v, idf)
        # print(11, len(train_feedback_info), len(LTR_train_feature), len(choose_feature), len(unlabel_feature))

        train_x_FV, train_y_FV = split_data.get_train_feature_matrix(train_feedback_info, choose_feature)
        test_feature = np.array(test_feature)

        y_predict = get_LTR_predict_LTR(test_feature, test_feedback_info, choose_feature, train_feedback_info)
        rank_mod, rankall = [], []
        for n in range(len(test_query)):
            temp_pred = y_predict[10 * n:10 * n + 10]
            pred, sum_pred = [], 0
            for i in range(10):
                sum_pred += temp_pred[i]+5
            for num in range(10):
                sum = (temp_pred[num]+5)/sum_pred
                pred.append(sum)
            rank_mod, rankall = metric.re_sort(pred, rec_api_test, test_answer, n, rank_mod, rankall)
        temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(rec_api_test))
        top1 += temp_top1
        top3 += temp_top3
        top5 += temp_top5
        map += temp_map
        mrr += temp_mrr
    print(top1/10, top3/10, top5/10, map/10, mrr/10)

    fw = open('../data/biker_metric_shareFR.csv', 'a+', newline='')
    writer = csv.writer(fw)
    writer.writerow(('ltr', num_choose, top1/10, top3/10, top5/10, map/10, mrr/10))
    fw.close()
