import csv


def re_sort(pred, rec_api_test, answer_test, n, rank_mod, rankall, rem = -10):
    rec_api = rec_api_test[10*n:10*n+10]
    sort, rec = [], []
    for i in range(10):
        sort.append(pred.index(max(pred)) + 1)
        rec.append(max(pred))
        pred[pred.index(max(pred))] = rem
    print(sort, rec_api)

    # 将api重新排序，输出相关结果
    rank_temp = -1
    rankall_temp = []
    api_temp = ''
    flag = 0
    for i in sort:
        api_mod = rec_api[i-1].lower()
        # api_mod = rec_api[i - 1]
        print(i, api_mod, answer_test[n])
        if api_mod in answer_test[n]:
            if flag == 0:
                rank_temp = sort.index(i) + 1
                api_temp = api_mod
                flag = 1
            rankall_temp.append(sort.index(i) + 1)

    # writer.writerow((rank_temp, test[n], api_temp))
    rank_mod.append(rank_temp)
    rankall.append(rankall_temp)
    print('rank:', rank_temp, 'original', sort[rank_temp-1])
    return rank_mod, rankall


def ALTR_re_sort(pred, rec_api_test, answer_test, n, rank_mod, rankall, rem = -10):
    rec_api = rec_api_test[10*n:10*n+10]
    sort, rec = [], []
    for i in range(10):
        sort.append(pred.index(max(pred)) + 1)
        rec.append(max(pred))
        pred[pred.index(max(pred))] = rem
    # print(sort, rec_api)

    # 将api重新排序，输出相关结果
    rank_temp = -1
    rankall_temp = []
    api_temp = ''
    flag = 0
    for i in sort:
        api_mod = rec_api[i-1].lower()
        # api_mod = rec_api[i - 1]
        # print(i, api_mod, answer_test[n])
        if api_mod in answer_test[n]:
            if flag == 0:
                rank_temp = sort.index(i) + 1
                api_temp = api_mod
                flag = 1
            rankall_temp.append(sort.index(i) + 1)

    # writer.writerow((rank_temp, test[n], api_temp))
    rank_mod.append(rank_temp)
    rankall.append(rankall_temp)
    # print('rank:', rank_temp, 'original', sort[rank_temp-1])
    return rank_mod, rankall


def metric_val(rank_mod, rankall, len_n):
    top1, top3, top5, map, mrr, miss = 0, 0, 0, 0, 0, 0
    for n in rank_mod:
        if n == 1:
            top1 += 1
        if n < 4 and n > 0:
            top3 += 1
        if n < 6 and n > 0:
            top5 += 1
        if n == -1:
            miss += 1
        else:
            mrr += 1/n

    for n in rankall:
        temp = 0
        count = 0
        count_miss = 0
        for i in range(len(n)):
            count += 1
            temp += count/n[i]
        if len(n) != 0:
            temp = temp/len(n)
        else:
            count_miss += 1
        map += temp
    print('top1', 10*top1/len_n)

    return 10*top1/len_n, 10*top3/len_n, 10*top5/len_n, 10*map/len_n, 10*mrr/len_n

# print('top1:', 10*top1 / len(answer_test))
# sum_1 += 10*top1 / len(y_test)
# sum_3 += 10*top3 / len(y_test)
# sum_5 += 10*top5 / len(y_test)
# sum_map += 10*map / len(y_test)
# sum_mrr += 10*mrr / len(y_test)