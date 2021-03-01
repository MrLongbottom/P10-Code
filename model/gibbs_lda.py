import re
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from preprocess.preprocessing import preprocessing


# def preprocessing(path: str):
#     documents, category = preprocessing(path)
#     word2id = {}
#     id2word = {}
#     docs = []
#     current_document = []
#     current_word_id = 0
#     for document in tqdm(list(documents.values())[:1000]):
#         segList = document.split(' ')
#         for word in segList:
#             word = word.lower().strip()
#             if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords.words('danish'):
#                 if word in word2id:
#                     current_document.append(word2id[word])
#                 else:
#                     current_document.append(current_word_id)
#                     word2id[word] = current_word_id
#                     id2word[current_word_id] = word
#                     current_word_id += 1
#         docs.append(current_document)
#         current_document = []
#     return docs, word2id, id2word


def gibbs(documents, doc_topic, topic_word):
    print("Random initialization")
    with Pool(processes=8) as p:
        max_ = len(documents)
        with tqdm(total=max_) as pbar:
            for score in p.imap(partial(sampling, doc_topic, topic_word), enumerate(documents)):
                pbar.update()


def sampling(doc_topic, topic_word, doc):
    d_index = doc[0]
    for w_index, word in enumerate(doc[1]):
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
    corpora, data = preprocessing("data/2017_data.json")
    data = [list(filter(lambda a: a != -1, corpora.doc2idx(doc))) for doc in data]
    N = len(data)
    M = len(corpora.token2id)
    document_topic_dist = np.zeros([N, K]) + alpha
    topic_word_dist = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta
    random_initialize(data, document_topic_dist, topic_word_dist)
    for i in tqdm(range(0, iterationNum)):
        gibbs(data, document_topic_dist, topic_word_dist)
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
