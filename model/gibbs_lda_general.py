import time
from typing import List

import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import get_coherence, mean_topic_diff, get_topics, _conditional_distribution, x_perplexity, \
    increase_count, decrease_count
from model.save import Model
from preprocess.preprocessing import prepro_file_load


def random_initialize(documents):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    feature_topic = np.zeros([num_feature, num_topics]) + alpha
    feature_topic_c = np.zeros([num_feature]) + num_topics * alpha
    topic_word = np.zeros([num_topics, W]) + beta
    topic_word_c = np.zeros([num_topics]) + W * beta
    wt_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        feature = doc2feature[d_index]
        for word in doc:
            pz = _conditional_distribution(feature, word, feature_topic, feature_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            curr_doc.append(topic)
            increase_count(feature, word, topic, feature_topic, feature_topic_c, topic_word, topic_word_c)
        wt_assignment.append(curr_doc)
    return wt_assignment, feature_topic, feature_topic_c, topic_word, topic_word_c


def gibbs_sampling(documents: List[np.ndarray],
                   feature_topic: np.ndarray,
                   feature_topic_c: np.ndarray,
                   topic_word: np.ndarray,
                   topic_word_c: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param feature_topic_c: the counts for authors
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_word_c: the number of the times each topic is used within words
    :param documents: a list of documents with their word ids
    :param feature_topic: a matrix describing the number of times each topic is used for each author
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        feature = doc2feature[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(feature, word, topic, feature_topic, feature_topic_c, topic_word, topic_word_c)

            # Sample a new topic based on author_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution(feature, word, feature_topic, feature_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(feature, word, topic, feature_topic, feature_topic_c, topic_word, topic_word_c)


if __name__ == '__main__':
    feature = "category"
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    doc2feature = prepro_file_load(f"doc2{feature}")
    num_feature = len(set(list(doc2feature.values())))
    doc2word = list(prepro_file_load("doc2word").items())
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())
    D, W = (dictionary.num_docs, len(dictionary))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, random_state=1337)

    word_topic_assignment, feature_topic, feature_topic_c, topic_word, topic_word_c = random_initialize(doc2word)

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, feature_topic, feature_topic_c, topic_word, topic_word_c, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              " Perplexity: ",
              x_perplexity(test_docs, feature_topic, feature_topic_c, topic_word, topic_word_c, doc2feature),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, num_topics, topic_word),
              " Topic Diff: ", mean_topic_diff(topic_word))
    model = Model(num_topics, alpha, beta, feature_topic, feature_topic_c, topic_word, topic_word_c, feature)
    model.save_model()
    print(get_topics(dictionary, num_topics, topic_word))
