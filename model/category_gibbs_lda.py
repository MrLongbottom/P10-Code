import pickle
import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import get_coherence, mean_topic_diff, get_topics, decrease_count, increase_count, \
    _conditional_distribution, cat_perplexity
from preprocess.preprocessing import prepro_file_load


def random_initialize(documents: List[np.ndarray], doc2category, num_categories: int, W: int, num_topics: int,
                      alpha: float, beta: float):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    category_topic = np.zeros([num_categories, num_topics]) + alpha
    topic_word = np.zeros([num_topics, W]) + beta
    word_topic_c = np.zeros([num_topics]) + W * beta
    cat_topic_c = np.zeros([num_categories]) + num_topics * alpha
    wt_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        cat = doc2category[d_index]
        for word in doc:
            pz = _conditional_distribution(cat, word, topic_word, category_topic, word_topic_c, cat_topic_c)
            topic = np.random.multinomial(1, pz / pz.sum()).argmax()
            curr_doc.append(topic)
            category_topic[cat, topic] += 1
            topic_word[topic, word] += 1
            word_topic_c[topic] += 1
            cat_topic_c[cat] += 1
        wt_assignment.append(curr_doc)
    return wt_assignment, category_topic, topic_word, word_topic_c, cat_topic_c


def gibbs_sampling_category(documents: List[np.ndarray], doc2category, 
                            cat_topic: np.ndarray,
                            topic_word: np.ndarray,
                            word_topic_count: np.ndarray,
                            doc_topic_count: np.ndarray,
                            word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_count: the number of the times each topic is used
    :param documents: a list of documents with their word ids
    :param cat_topic: a matrix describing the number of times each topic within each category
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        c_index = doc2category[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(topic, topic_word, cat_topic, c_index, word, word_topic_count, doc_topic_count)

            # Sample a new topic based on cat_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution(c_index, word, topic_word, cat_topic, word_topic_count, doc_topic_count)
            topic = np.random.multinomial(1, pz).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(topic, topic_word, cat_topic, c_index, word, word_topic_count, doc_topic_count)


def setup_category(alpha: float, beta: float, num_topics: int):
    doc2word = list(prepro_file_load("doc2word").items())
    dictionary = prepro_file_load('corpora')
    doc2category = prepro_file_load("doc2category")
    num_categories = len(doc2category)
    D, W = (dictionary.num_docs, len(dictionary))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33)

    word_topic_assignment, category_topic_dist, topic_word_dist, word_topic_count, cat_topic_count = random_initialize(
        doc2word, doc2category, num_categories, W, num_topics, alpha, beta)
    return train_docs, test_docs, doc2category, word_topic_assignment, category_topic_dist, topic_word_dist, word_topic_count, cat_topic_count


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    doc2category = prepro_file_load("doc2category")
    num_categories = len(set(list(doc2category.values())))
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    doc2word = list(prepro_file_load("doc2word").items())
    D, W = (corpora.num_docs, len(corpora))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, shuffle=True)

    word_topic_assignment, category_topic_dist, topic_word_dist, wt_count, dt_count = random_initialize(doc2word)

    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling_category(train_docs, category_topic_dist, topic_word_dist, wt_count, dt_count, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              " Perplexity: ", cat_perplexity(test_docs, category_topic_dist, topic_word_dist, wt_count, dt_count),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, corpora, num_topics, topic_word_dist),
              " Topic Diff: ", mean_topic_diff(topic_word_dist))
    print(get_topics(corpora, num_topics, topic_word_dist))
