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


def perplexity(documents: List[np.ndarray],  document_topic, document_topic_count, topic_word, topic_word_c) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    n = 0
    ll = 0.0
    for d, doc in documents:
        for w in doc:
            ll += np.log(((topic_word[:, w] / topic_word_c) * (document_topic[d, :] / document_topic_count[d])).sum())
            n += 1
    return np.exp(ll / (-n))


def x_perplexity(documents: List[np.ndarray], feature_topic, feature_topic_c, topic_word, topic_word_c, doc2feature) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    n = 0
    ll = 0.0
    for d, doc in documents:
        feature = doc2feature[d]
        for w in doc:
            ll += np.log(((topic_word[:, w] / topic_word_c) * (feature_topic[feature, :] / feature_topic_c[feature])).sum())
            n += 1
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


def increase_count(feature1, word, topic, feature1_topic, feature1_topic_c, topic_word, topic_word_c):
    feature1_topic[feature1, topic] += 1
    feature1_topic_c[feature1] += 1
    topic_word[topic, word] += 1
    topic_word_c[topic] += 1


def decrease_count(feature1, word, topic, feature1_topic, feature1_topic_c, topic_word, topic_word_c):
    feature1_topic[feature1, topic] -= 1
    feature1_topic_c[feature1] -= 1
    topic_word[topic, word] -= 1
    topic_word_c[topic] -= 1


def _conditional_distribution(doc_index, word, doc_topic, doc_topic_count, topic_word, topic_word_count):
    left = np.divide(topic_word[:, word], topic_word_count)
    right = np.divide(doc_topic[doc_index, :], doc_topic_count[doc_index])
    p_z = np.multiply(left, right)
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z


def _conditional_distribution_combination(author, category, word, author_topic, author_topic_count,
                                          category_topic, category_topic_count, topic_word, topic_word_count):
    left = np.divide(topic_word[:, word], topic_word_count)
    middle = np.divide(category_topic[category, :], category_topic_count[category])
    right = np.divide(author_topic[author, :], author_topic_count[author])
    p_z = np.multiply(left, middle, right)
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z
