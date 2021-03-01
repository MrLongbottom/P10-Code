import itertools
import pickle
import time
from functools import partial
from multiprocessing import Pool
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from preprocess.preprocessing import preprocessing


def gibbs(documents, doc_topic, topic_word):
    with Pool(processes=8) as p:
        max_ = len(documents)
        with tqdm(total=max_) as pbar:
            for score in p.imap(partial(sampling, doc_topic, topic_word), sets):
                pbar.update()


def sampling(doc_topic, topic_word, documents):
    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            topic = Z[d_index][w_index]
            doc_topic[d_index, topic] -= 1
            topic_word[topic, word] -= 1
            nz[topic] -= 1
            pz = np.divide(np.multiply(doc_topic[d_index, :], topic_word[:, word]), nz)
            topic = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d_index][w_index] = topic
            doc_topic[d_index, topic] += 1
            topic_word[topic, word] += 1
            nz[topic] += 1


def random_initialize(documents, doc_topic, topic_word):
    print("Random Initilization")
    for d, doc in tqdm(enumerate(documents)):
        zCurrentDoc = []
        for w in doc:
            pz = np.divide(np.multiply(doc_topic[d, :], topic_word[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            doc_topic[d, z] += 1
            topic_word[z, w] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)


# def gibbsSampling(documents, doc_topic, topic_word):
#     for d_index, doc in enumerate(documents):
#         for w_index, word in enumerate(doc):
#             topic = Z[d_index][w_index]
#             doc_topic[d_index, topic] -= 1
#             topic_word[topic, word] -= 1
#             nz[topic] -= 1
#             pz = np.divide(np.multiply(doc_topic[d_index, :], topic_word[:, word]), nz)
#             topic = np.random.multinomial(1, pz / pz.sum()).argmax()
#             Z[d_index][w_index] = topic
#             doc_topic[d_index, topic] += 1
#             topic_word[topic, word] += 1
#             nz[topic] += 1


def perplexity(data):
    nd = np.sum(document_topic_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(data):
        for w in doc:
            ll = ll + np.log(((topic_word_dist[:, w] / nz) * (document_topic_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    Z = []
    K = 10
    with open("../preprocess/generated_files/doc_word_matrix", 'rb') as file:
        doc_word_matrix = pickle.load(file)
    doc_word_matrix = np.array(doc_word_matrix.to_dense(), dtype=int)
    N = doc_word_matrix.shape[0]
    M = doc_word_matrix.shape[1]
    doc_word_matrix = [np.nonzero(x)[0] for x in doc_word_matrix]
    sets = [list(x) for x in chunks(list(enumerate(doc_word_matrix)), int(len(doc_word_matrix) / 8))]

    document_topic_dist = np.zeros([N, K]) + alpha
    topic_word_dist = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta
    random_initialize(doc_word_matrix, document_topic_dist, topic_word_dist)
    for i in tqdm(range(0, iterationNum)):
        gibbs(doc_word_matrix, document_topic_dist, topic_word_dist)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(doc_word_matrix))

    topic_words = []
    maxTopicWordsNum = 10
    for z in range(0, K):
        ids = topic_word_dist[z, :].argsort()
        topic_word = []
        for j in ids:
            topic_word.insert(0, corpora.id2token[j])
        topic_words.append(topic_word[0: min(10, len(topic_word))])
    print(topic_words)
