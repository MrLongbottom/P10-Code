import re
import time

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

from loading import load_document_file


def preprocessing(path: str):
    documents, category = load_document_file(path)
    word2id = {}
    id2word = {}
    docs = []
    current_document = []
    current_word_id = 0
    for document in tqdm(list(documents.values())[:1000]):
        segList = document.split(' ')
        for word in segList:
            word = word.lower().strip()
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords.words('danish'):
                if word in word2id:
                    current_document.append(word2id[word])
                else:
                    current_document.append(current_word_id)
                    word2id[word] = current_word_id
                    id2word[current_word_id] = word
                    current_word_id += 1
        docs.append(current_document)
        current_document = []
    return docs, word2id, id2word


def random_initialize(documents, doc_topic, topic_word):
    for d, doc in enumerate(documents):
        zCurrentDoc = []
        for w in doc:
            pz = np.divide(np.multiply(doc_topic[d, :], topic_word[:, w]), topics)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            doc_topic[d, z] += 1
            topic_word[z, w] += 1
            topics[z] += 1
        Z.append(zCurrentDoc)


def gibbs_sampling(documents, doc_topic, topic_word, tops):
    for d_index, doc in enumerate(documents):
        for w_index, word in enumerate(doc):
            topic = Z[d_index][w_index]
            doc_topic[d_index, topic] -= 1
            topic_word[topic, word] -= 1
            tops[topic] -= 1

            probability_of_topic = np.divide(np.multiply(doc_topic[d_index, :], topic_word[:, word]), topics)
            drawn_topic = np.random.multinomial(1, probability_of_topic / probability_of_topic.sum()).argmax()

            Z[d_index][w_index] = drawn_topic
            doc_topic[d_index, drawn_topic] += 1
            topic_word[drawn_topic, word] += 1
            tops[drawn_topic] += 1
    return doc_topic, topic_word, tops


def perplexity():
    nd = np.sum(document_topic_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for w in doc:
            ll = ll + np.log(((topic_word_dist[:, w] / topics) * (document_topic_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    Z = []
    K = 10
    docs, word2id, id2word = preprocessing("data/2017_data.json")
    N = len(docs)
    M = len(word2id)
    document_topic_dist = np.zeros([N, K]) + alpha
    topic_word_dist = np.zeros([K, M]) + beta
    topics = np.zeros([K]) + M * beta
    random_initialize(docs, document_topic_dist, topic_word_dist)
    for i in range(0, iterationNum):
        document_topic_dist, topic_word_dist, topics = gibbs_sampling(docs, document_topic_dist, topic_word_dist, topics)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity())

    topic_words = []
    maxTopicWordsNum = 10
    for z in range(0, K):
        ids = topic_word_dist[z, :].argsort()
        tword = []
        for j in ids:
            tword.insert(0, id2word[j])
        topic_words.append(tword[0: min(10, len(tword))])
    print(topic_words)
