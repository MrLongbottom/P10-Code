import itertools
import time
from functools import partial
from multiprocessing import Pool
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from preprocess.preprocessing import preprocessing


# def gibbs(documents, doc_topic, topic_word):
#     with Pool(processes=8) as p:
#         max_ = len(documents)
#         with tqdm(total=max_) as pbar:
#             for score in p.imap(partial(sampling, doc_topic, topic_word), enumerate(documents)):
#                 pbar.update()
#
#
# def sampling(doc_topic, topic_word, doc):
#     d_index = doc[0]
#     for w_index, word in enumerate(doc[1]):
#         topic = Z[d_index][w_index]
#         doc_topic[d_index, topic] -= 1
#         topic_word[topic, word] -= 1
#         nz[topic] -= 1
#         pz = np.divide(np.multiply(doc_topic[d_index, :], topic_word[:, word]), nz)
#         topic = np.random.multinomial(1, pz / pz.sum()).argmax()
#         Z[d_index][w_index] = topic
#         doc_topic[d_index, topic] += 1
#         topic_word[topic, word] += 1
#         nz[topic] += 1


def random_initialize(documents, doc_topic, topic_word):
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


def gibbsSampling(documents, doc_topic, topic_word):
    for d_index, doc in enumerate(documents):
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


def perplexity(data):
    nd = np.sum(document_topic_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(data):
        for w in doc:
            ll = ll + np.log(((topic_word_dist[:, w] / nz) * (document_topic_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    Z = []
    K = 10
    corpora, documents = preprocessing("data/2017_data.json")
    N = len(documents)
    M = len(corpora.token2id)
    data = [list(filter(lambda a: a != -1, corpora.doc2idx(doc))) for doc in documents]

    document_topic_dist = np.zeros([N, K]) + alpha
    topic_word_dist = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta
    random_initialize(data, document_topic_dist, topic_word_dist)
    for i in tqdm(range(0, iterationNum)):
        gibbsSampling(data, document_topic_dist, topic_word_dist)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(data))

    topic_words = []
    maxTopicWordsNum = 10
    for z in range(0, K):
        ids = topic_word_dist[z, :].argsort()
        topic_word = []
        for j in ids:
            topic_word.insert(0, corpora.id2token[j])
        topic_words.append(topic_word[0: min(10, len(topic_word))])
    print(topic_words)
