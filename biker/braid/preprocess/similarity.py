from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import gensim
import _pickle as pickle
import numpy as np
from sklearn import preprocessing
import math


def init_doc_matrix(doc,w2v):

    matrix = np.zeros((len(doc),100)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[word])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))

    return matrix


def init_w2a_matrix(doc, w2a_dict):
    matrix = np.zeros((len(doc), 100)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2a_dict:
            matrix[i] = np.array(w2a_dict[word])
            # print('matrix',matrix[i])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))

    return matrix


def init_doc_idf_vector(doc,idf):
    idf_vector = np.zeros((1,len(doc)))  # word embedding size is 100
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word][1]

    return idf_vector


def sim_doc_pair(matrix1,matrix2,idf1,idf2):

    sim12 = (idf1*(matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()

    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()

    return (sim12 + sim21)/2


