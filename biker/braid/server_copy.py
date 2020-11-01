import csv
import json
import time

from websocket_server import WebsocketServer

from preprocess import recommendation
from preprocess import similarity
import gensim
import _pickle as pickle
import time
import xgboost as xgb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import split_data, feedback, metric, braid_LTR, braid_AL
import warnings
warnings.filterwarnings("ignore")

from flask import Flask
from flask import jsonify
from flask_cors import CORS



print("Now, start to load data.")
w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding
idf = pickle.load(open('../data/idf', 'rb')) # pre-trained idf value of all words in the w2v dictionary
questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = dict() # In online mode, there is no need to remove duplicate question of the query

print('loading data finished')

rerank = []
responseToClient = []
g_query_str = ''
feedback_inf = None
api_feature = None
sort = []

def text2feat(api, api_descriptions, w2v, idf, query_matrix, query_idf_vector):
    api_matrix, api_idf_vector = feedback.load_matrix(api, w2v, idf)
    api_descriptions_matrix, api_descriptions_idf_vector = feedback.load_matrix(api_descriptions, w2v, idf)

    # 获取api及doc信息并计算其相似度，相关问题在推荐中已经获得
    api_sim = similarity.sim_doc_pair(query_matrix, api_matrix, query_idf_vector, api_idf_vector)
    if api_descriptions == 'null':
        api_desc_sim = 0
    else:
        api_desc_sim = similarity.sim_doc_pair(query_matrix, api_descriptions_matrix, query_idf_vector, api_descriptions_idf_vector)

    # 将获得信息按api为一列放入sum_inf中
    sum_inf = list()
    sum_inf.append(api_sim)
    sum_inf.append(api_desc_sim)

    # 将所有特征封装成字典并返回，这样得到特征之后能直接输出topn的相关特征
    api_inf = dict()
    api_desc_inf = dict()
    api_inf[api] = api_sim
    api_desc_inf[api_descriptions] = api_desc_sim

    return sum_inf, api_inf, api_desc_inf

def get_AL_predict(test_feature, choose_feature, unlabel_feature, test_query, choose_query, choose_answer, unlabel_query, unlabel_answer, rec_api_test, rec_api_choose, rec_api_unlabel, w2v, idf):
    unlabel_feedback_info = feedback.get_feedback_inf(unlabel_query, choose_query, choose_answer, rec_api_unlabel, w2v, idf)
    label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    X_train, y_train = braid_AL.get_active_data(unlabel_feedback_info, unlabel_feature)
    X_feedback, y_feedback = braid_AL.get_active_data(label_feedback_info, choose_feature)

    # initializing the active learner
    learner = ActiveLearner(
        # estimator=KNeighborsClassifier(n_neighbors=4),
        estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        X_training=X_feedback, y_training=y_feedback
    )

    predict, sel_query, add_unlabel_feature = [], [], []
    if len(unlabel_query) > 0:
        # pool-based sampling
        n_queries = 40
        sel_idx, sel_label = [], []
        for idx in range(n_queries):
            query_idx, query_instance = uncertainty_sampling(classifier=learner, X=X_train)
            idx = int(query_idx/10)
            learner.teach(
                X=X_train[query_idx].reshape(1, -1),
                y=y_train[query_idx].reshape(1, )
            )

            # add queried instance into FR
            choose_query.append(unlabel_query[idx])
            choose_answer.append(unlabel_answer[idx])
            rec_api_choose.extend(rec_api_unlabel[idx*10:idx*10+10])
            choose_feature.extend(unlabel_feature[idx*10:idx*10+10])

            # remove queried instance from pool
            for i in range(10):
                X_train = np.delete(X_train, idx*10, axis=0)
                y_train = np.delete(y_train, idx*10)
            del unlabel_query[idx]
            del unlabel_answer[idx]
            del rec_api_unlabel[idx*10:idx*10+10]
            del unlabel_feature[idx*10:idx*10+10]
            if len(X_train) == 0:
                break

    add_label_feedback_info = feedback.get_feedback_inf(choose_query, choose_query, choose_answer, rec_api_choose, w2v, idf)
    new_X_feedback, new_y_feedback = braid_AL.get_active_data(add_label_feedback_info, choose_feature)
    learner = ActiveLearner(
        # estimator=KNeighborsClassifier(n_neighbors=4),
        estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        X_training=new_X_feedback, y_training=new_y_feedback
    )
    feedback_info = feedback.get_feedback_inf(test_query, choose_query, choose_answer, rec_api_test, w2v, idf)
    X = split_data.get_test_feature_matrix(feedback_info, test_feature)

    X_test = np.array(X)
    # 用反馈数据学习过后的模型来预测测试数据
    for query_idx in range(10):
        y_pre = learner.predict_proba(X=X_test[query_idx].reshape(1, -1))
        predict.append(float(y_pre[0, 1]))
    # print(predict)
    # print('new_choose', len(choose_query), len(choose_answer))

    return predict, X, new_X_feedback, new_y_feedback


# fr = open('../data/10_query_1.txt', 'r')
# time_query = []
# for row in fr.readlines():
#     time_query.append(row.split('\n')[0])
#     print(row.split('\n')[0])
# for query in time_query:



def process_input(msg='how to convert int to string?'):
    query = msg
    query_matrix, query_idf_vector = feedback.load_matrix(query, w2v, idf)

    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
    # recommended_api = recommendation.recommend_api_class(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_classes,-1)    

    # combine api_relevant feature with FF
    pos = -1
    rec_api = []
    api_dict_desc = {}
    x, api_feature = [], []
    for i,api in enumerate(recommended_api):
        # print('Rank',i+1,':',api)
        rec_api.append(api)
        # recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
        api_descriptions, questions_titles = recommendation.summarize_api_method(api, top_questions, questions, javadoc,
                                                                                 javadoc_dict_methods)
        
        api_dict_desc[api] = api_descriptions

        sum_inf, api_inf, api_desc_inf = text2feat(api, api_descriptions, w2v, idf, query_matrix, query_idf_vector)
        api_feature.append(sum_inf)
        # print(api_feature)
        if i == 9:
            break
    # print('##################')

    start1 = time.time()
    # feedback info of user query from SO
    fr = open('../data/feedback_all.csv', 'r')
    reader = csv.reader(fr)
    so_query, so_answer = [], []
    for row in reader:
        so_query.append(row[0])
        so_answer.append(row[1:])

    # feedback info of user query from FR
    fr = open('../data/feedback_rec.csv', 'r')
    reader = csv.reader(fr)
    choose_query, choose_answer = [], []
    for row in reader:
        choose_query.append(row[0])
        choose_answer.append(row[1:])
    # print('choose', choose_query, choose_answer)
    # print(query)
    feedback_inf = feedback.get_feedback_inf(query, choose_query, choose_answer, rec_api, w2v, idf)

    # print(len(choose_query), len(choose_answer), len(api_feature))
    # FV = RF+FF
    for i in range(len(api_feature)):
        sum = api_feature[i]
        sum.extend(feedback_inf[i])
        x.append(sum)
    while len(x)%10:
        x.append([0, 0, 0, 0, 0, 0, 0])
        rec_api.append('null')

    # feature info of FR
    fr = open('../data/feedback_feature_rec.csv', 'r')
    reader = csv.reader(fr)
    y_feature, x_feautre, api_relevant_feature, rec_api_choose = [], [], [], []
    for row in reader:
        # y_feature.append(row[0])
        x_feautre.append(row[:-1])
        api_relevant_feature.append(row[1:3])
        rec_api_choose.append(row[-1])

    #feature info of SO
    fr = open('../data/get_feature_method.csv', 'r')
    reader = csv.reader(fr)
    unlabel_feature, rec_api_unlabel = [], []
    for row in reader:
        # y_feature.append(row[0])
        unlabel_feature.append(row[:-1])
        rec_api_unlabel.append(row[-1])

    start2 = time.time()

    # AL_choose_feature, AL_unlabel_feature = split_data.get_choose(AL_train_feature, choose)
    pred2, add_x_FR, add_x_FV, add_y_FV = get_AL_predict(x, x_feautre, unlabel_feature, query, choose_query, choose_answer, so_query, so_answer, rec_api, rec_api_choose, rec_api_unlabel, w2v, idf)

    start3 = time.time()
    pred1 = braid_LTR.get_LTR_predict(add_x_FR, add_x_FV, add_y_FV)

    rem = -10
    start4 = time.time()
    rec, rec_LTR, rec_AL = [], [], []
    sort, sort_LTR, sort_AL = [], [], []
    pred = []
    sum_pred1, sum_pred2 = 0, 0
    for i in range(10):
        sum_pred1 += pred1[i]+5
        sum_pred2 += pred2[i]
    al_idx = []
    rerank_al = sorted(pred2, reverse=True)
    for i in range(10):
        temp = rerank_al.index(pred2[i])+1
        while temp in al_idx:
            temp += 1
        al_idx.append(temp)
    # print(al_idx)
    m = 0.6
    for num in range(10):
        sum = (pred1[num]+5)/10 + m*pred2[num]/al_idx[num]
        pred.append(sum)
    # print('LTR', pred1)
    # print('AL', pred2)
    # print('LTR+AL', pred)

    for i in range(10):
        sort.append(pred.index(max(pred)) + 1)
        sort_LTR.append(pred1.index(max(pred1)) + 1)
        sort_AL.append(pred2.index(max(pred2)) + 1)
        rec.append(max(pred))
        rec_LTR.append(max(pred1))
        rec_AL.append(max(pred2))
        pred[pred.index(max(pred))] = rem
        pred1[pred1.index(max(pred1))] = rem
        pred2[pred2.index(max(pred2))] = rem
    # print(sort, rec_api)

    # 将api重新排序，输出相关结果
    
    for i in sort:
        api_mod = rec_api[i-1]
        print(sort.index(i) + 1, api_mod)
        api_obj = {'id':sort.index(i) + 1, 'api':api_mod, 'desc':api_dict_desc[api_mod] }
        rerank.append(api_mod)
        responseToClient.append(api_obj)
    start5 = time.time()
    print(type(rerank))
    print(json.dumps(rerank))
    print(rerank)
    print(responseToClient)
    return responseToClient
    
def feedback_rec_func(choose):

    if int(choose):
        fw = open('../data/feedback_rec.csv', 'a+', newline='')
        writer = csv.writer(fw)
        writer.writerow((query, rerank[int(choose)-1]))
        fw.close()

        fw = open('../data/feedback_feature_rec.csv', 'a+', newline='')
        writer = csv.writer(fw)
        for i in sort:
            y = 0
            if sort.index(i) == int(choose)-1:
                y = 1
                feedback_inf[i-1][0] = 1
            writer.writerow([y] + api_feature[i-1][:2] + feedback_inf[i-1] + [rerank[sort.index(i)]])
        fw.close()
        print(query, rerank[int(choose)-1])
        return {'success':True, 'msg': "反馈已处理"}
    else:
        print('none')
        return {'success':False,'msg': '参数不是整数'}

# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])
    

def message_received(client, server, message):
    print('client',type(client), client)
    print('server',type(server), server)
    print('message',type(message), message)

    res = process_input(message)
    
    if len(message) > 200:
        message = message[:200] + '...'
    print("Client(%d) said: %s" % (client['id'], message))

    server.send_message(client, json.dumps(res))

# PORT = 8765
# server = WebsocketServer(PORT)
# print('websocket server started!running at port {}'.format(PORT))
# server.set_fn_new_client(new_client)
# server.set_fn_client_left(client_left)
# server.set_fn_message_received(message_received)
# server.run_forever()

print("开启flask")
app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/query/<query_str>')
def query_rec(query_str):
    global g_query_str
    g_query_str = query_str
    return jsonify(process_input(query_str)),200,[('Access-Control-Allow-Origin','*')]

@app.route('/feedback/<choose>')
def query(choose):
    return jsonify(feedback_rec_func(int(choose))),200,[('Access-Control-Allow-Origin','*')]

app.run()
print('flask 开启成功')