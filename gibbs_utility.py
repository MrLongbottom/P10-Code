from typing import List

from gensim.models import CoherenceModel
from scipy.spatial import distance

import numpy as np

# supported measures = {'u_mass', 'c_v', 'c_uci', 'c_npmi'}
from preprocess.preprocessing import prepro_file_load


def coherence(topics, doc2bow, dictionary, texts, coherence_measure='c_v'):
    cm = CoherenceModel(topics=topics, corpus=doc2bow, dictionary=dictionary, texts=texts, coherence=coherence_measure)
    return cm.get_coherence()


def topic_diff(topic_word_dist):
    topic_difference_matrix = np.zeros((topic_word_dist.shape[0] - 1, topic_word_dist.shape[0] - 1))
    for topics in np.ndindex(topic_difference_matrix.shape):
        # don't compare topics to themselves
        if topics[0] == topics[1]:
            continue
        # fill out distance matrix with distance between topic1 and topic2
        topic_difference_matrix[topics] = distance.jensenshannon(topic_word_dist[topics[0]], topic_word_dist[topics[1]])
    return topic_difference_matrix


def mean_topic_diff(topic_word_dist):
    td = topic_diff(topic_word_dist)
    return np.mean(td)


def perplexity(documents: List[np.ndarray], dt_dist, tw_dist, topic_c) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    nd = np.sum(dt_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(documents):
        for w in doc:
            ll = ll + np.log(((tw_dist[:, w] / topic_c) * (dt_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def cat_perplexity(documents: List[np.ndarray], ct_dist, tw_dist, topic_c) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    nd = np.sum(ct_dist, 1)
    n = 0
    ll = 0.0
    doc2category = prepro_file_load("doc2category")
    for d, doc in enumerate(documents):
        cat = doc2category[d]
        for w in doc:
            ll = ll + np.log(((tw_dist[:, w] / topic_c) * (ct_dist[cat, :] / nd[cat])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def get_topics(corpora, num_topics, topic_word_dist, num_of_word_per_topic: int = 10):
    """
    Looks at the topic word distribution and sorts each topic based on the word count
    :param num_of_word_per_topic: how many word is printed within each topic
    :return: print the topics
    """
    topic_words = []
    id2token = {v: k for k, v in corpora.token2id.items()}
    for z in range(0, num_topics):
        ids = topic_word_dist[z, :].argsort()
        topic_word = []
        for j in ids:
            topic_word.insert(0, id2token[j])
        topic_words.append(topic_word[0: min(num_of_word_per_topic, len(topic_word))])
    return topic_words


def get_coherence(doc2bow, dictionary, texts, corpora, num_topics, topic_word_dist):
    return coherence(topics=get_topics(corpora, num_topics, topic_word_dist), doc2bow=doc2bow, dictionary=dictionary,
                     texts=texts)


def increase_count(topic, topic_word, doc_topic, d_index, word, t_count):
    doc_topic[d_index, topic] += 1
    topic_word[topic, word] += 1
    t_count[topic] += 1


def decrease_count(topic, topic_word, doc_topic, d_index, word, t_count):
    doc_topic[d_index, topic] -= 1
    topic_word[topic, word] -= 1
    t_count[topic] -= 1
