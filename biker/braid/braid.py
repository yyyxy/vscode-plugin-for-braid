import split_data, feedback, metric, braid_LTR, braid_AL
import gensim
import _pickle as pickle
import csv
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()

w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')  # pre-trained word embedding
idf = pickle.load(open('../data/idf', 'rb'))  # pre-trained idf value of all words in the w2v dictionary

test_query, test_answer, train_query, train_answer, test_feature, train_feature, rec_api_test, rec_api_train = split_data.get_test_train()
LTR_train_feature = braid_LTR.get_LTR_feature(train_answer, rec_api_train, train_feature)
AL_train_feature = braid_AL.get_AL_feature(train_answer, rec_api_train, train_feature)
num_choose = 336
top1, top3, top5, map, mrr = 0, 0, 0, 0, 0
LTR_top1, LTR_top3, LTR_top5, LTR_map, LTR_mrr = 0, 0, 0, 0, 0
AL_top1, AL_top3, AL_top5, AL_map, AL_mrr = 0, 0, 0, 0, 0
# iteration begin
for round in range(10):
    choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_choose_unlabel(
        train_query, train_answer, rec_api_train, num_choose)
    # choose_query, choose_answer, rec_api_choose, unlabel_query, unlabel_answer, rec_api_unlabel, choose = split_data.split_10_choose_unlabel(
    #     train_query, train_answer, rec_api_train)

    AL_choose_feature, AL_unlabel_feature = split_data.get_choose(AL_train_feature, choose)
    AL_predict, add_x_FR, add_x_FV, add_y_FV = braid_AL.get_AL_predict(test_feature, AL_choose_feature, AL_unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_test, rec_api_choose, rec_api_unlabel, w2v, idf)

    # add_rec_api_choose = split_data.get_add_FR_rec_api(unlabel_query, rec_api_unlabel, add_unlabel_index)
    # rec_api_choose.extend(add_rec_api_choose)
    LTR_predict = braid_LTR.get_LTR_predict(add_x_FR, add_x_FV, add_y_FV)

    rank_mod, rankall, LTR_rank_mod, LTR_rankall, AL_rank_mod, AL_rankall = [], [], [], [], [], []
    m = 0.6
    for n in range(len(test_query)):
        pred1 = LTR_predict[10*n:10*n+10]
        pred2 = AL_predict[10*n:10*n+10]
        pred, sum_pred1,sum_pred2 = [], 0, 0
        LTR_pred, AL_pred, LTR_sum,AL_sum = [], [], 0, 0
        for i in range(10):
            sum_pred1 += pred1[i] + 5
            sum_pred2 += pred2[i]
        al_idx = []
        rerank_al = sorted(pred2, reverse=True)
        for i in range(10):
            temp = rerank_al.index(pred2[i])+1
            while temp in al_idx:
                temp += 1
            al_idx.append(temp)
        print(al_idx)
        for num in range(10):
            sum = m*(pred1[num]+5)/10+(1-m)*pred2[num]/al_idx[num]
            LTR_sum = m*(pred1[num]+5)/10
            AL_sum = (1-m)*pred2[num]/al_idx[num]
            pred.append(sum)
            LTR_pred.append(LTR_sum)
            AL_pred.append(AL_sum)
        print(LTR_pred)
        print(AL_pred)
        rank_mod, rankall = metric.re_sort(pred, rec_api_test, test_answer, n, rank_mod, rankall)
        LTR_rank_mod, LTR_rankall = metric.ALTR_re_sort(LTR_pred, rec_api_test, test_answer, n, LTR_rank_mod, LTR_rankall)
        AL_rank_mod, AL_rankall = metric.ALTR_re_sort(AL_pred, rec_api_test, test_answer, n, AL_rank_mod, AL_rankall)
    temp_top1, temp_top3, temp_top5, temp_map, temp_mrr = metric.metric_val(rank_mod, rankall, len(rec_api_test))
    LTR_temp_top1, LTR_temp_top3, LTR_temp_top5, LTR_temp_map, LTR_temp_mrr = metric.metric_val(LTR_rank_mod, LTR_rankall, len(rec_api_test))
    AL_temp_top1, AL_temp_top3, AL_temp_top5, AL_temp_map, AL_temp_mrr = metric.metric_val(AL_rank_mod, AL_rankall, len(rec_api_test))
    top1 += temp_top1
    top3 += temp_top3
    top5 += temp_top5
    map += temp_map
    mrr += temp_mrr
    LTR_top1 += LTR_temp_top1
    LTR_top3 += LTR_temp_top3
    LTR_top5 += LTR_temp_top5
    LTR_map += LTR_temp_map
    LTR_mrr += LTR_temp_mrr
    AL_top1 += AL_temp_top1
    AL_top3 += AL_temp_top3
    AL_top5 += AL_temp_top5
    AL_map += AL_temp_map
    AL_mrr += AL_temp_mrr
print(top1/10, top3/10, top5/10, map/10, mrr/10)
print(LTR_top1/10, LTR_top3/10, LTR_top5/10, LTR_map/10, LTR_mrr/10)
print(AL_top1/10, AL_top3/10, AL_top5/10, AL_map/10, AL_mrr/10)

fw = open('../data/biker_metric_shareFR.csv', 'a+', newline='')
writer = csv.writer(fw)
writer.writerow(('BRAID', num_choose, top1/10, top3/10, top5/10, map/10, mrr/10))
writer.writerow(('LTR', num_choose, LTR_top1/10, LTR_top3/10, LTR_top5/10, LTR_map/10, LTR_mrr/10))
writer.writerow(('AL', num_choose, AL_top1/10, AL_top3/10, AL_top5/10, AL_map/10, AL_mrr/10))
fw.close()

end = time.time()
print(end-start)
