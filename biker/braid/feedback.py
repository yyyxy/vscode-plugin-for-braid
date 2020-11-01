from preprocess import similarity
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer


def load_matrix(query, w2v, idf):
    # print(query)
    query_words = WordPunctTokenizer().tokenize(query.lower())
    if query_words[-1] == '?':
        query_words = query_words[:-1]
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    return query_matrix, query_idf_vector


def get_sim_query(train, test, w2v, idf):
    sim = 0
    for i in range(len(train)):
        train_matrix, train_idf = load_matrix(train, w2v, idf)
        test_matrix, test_idf = load_matrix(test, w2v, idf)
        sim = similarity.sim_doc_pair(train_matrix, test_matrix, train_idf, test_idf)
    return sim


def get_feedback_api(query, answer, query_matrix, query_idf_vector, w2v, idf):
    line = 0
    feeds = []
    for row in answer:
        if line > 0:
            question_matrix, question_idf_vector = load_matrix(query[answer.index(row)], w2v, idf)
            sim = similarity.sim_doc_pair(query_matrix, question_matrix, query_idf_vector, question_idf_vector)
            # 若query与反馈的问题相似，则将反馈问题的api信息加入
            if sim > 0.65:
                for n in range(len(row)):
                    feed = [query[answer.index(row)], row[n], sim]
                    feeds.append(feed)
        line += 1
    feeds = sorted(feeds, key=lambda item: item[2], reverse=True)
    while len(feeds) < 5:
        feeds.append([0, 0, 0])
    feed_sim = []
    for inf in feeds:
        if len(feed_sim) < 5:
            feed_sim.append(inf[2])
    return feeds, feed_sim


def get_feedback_inf(test, question, answer, rec_api_test, w2v, idf):
    feedback_inf = []
    length = len(rec_api_test)

    if isinstance(test, str):
        test = [test]

    for query in test:
        query_matrix, query_idf_vector = load_matrix(query, w2v, idf)
        feedbacks_inf, feed_sim = get_feedback_api(question, answer, query_matrix, query_idf_vector, w2v, idf)
        for api in rec_api_test[length*test.index(query):length*test.index(query)+length]:
            feed_label = []
            for feed in feedbacks_inf[0:5]:
                if feed[1] == api:
                    feed_label.append(feed[2])
                else:
                    feed_label.append(0)
            feedback_inf.append(feed_label)
    return feedback_inf
