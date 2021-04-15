import pickle
import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import increase_count, decrease_count, perplexity, get_coherence, get_topics, \
    _conditional_distribution
from model.save import Model
from preprocess.preprocessing import prepro_file_load


def random_initialize(documents: List[np.ndarray]):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    doc_topic = np.zeros([D, num_topics]) + alpha
    doc_topic_c = np.zeros([D]) + num_topics * alpha
    topic_word = np.zeros([num_topics, W]) + beta
    topic_word_c = np.zeros([num_topics]) + W * beta
    word_topic_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        for word in doc:
            pz = _conditional_distribution(d_index, word, doc_topic, doc_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            curr_doc.append(topic)
            increase_count(d_index, word, topic, doc_topic, doc_topic_c, topic_word, topic_word_c)
        word_topic_assignment.append(curr_doc)
    return word_topic_assignment, doc_topic, doc_topic_c, topic_word, topic_word_c


def gibbs_sampling(documents: List[np.ndarray],
                   doc_topic: np.ndarray,
                   doc_topic_c: np.ndarray,
                   topic_word: np.ndarray,
                   topic_word_c: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param doc_topic_count:
    :param topic_word_c:
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param documents: a list of documents with their word ids
    :param doc_topic: a matrix describing the number of times each topic within each document
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(d_index, word, topic, doc_topic, doc_topic_c, topic_word, topic_word_c)

            # Sample a new topic based on doc_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution(d_index, word, doc_topic, doc_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(d_index, word, topic, doc_topic, doc_topic_c, topic_word, topic_word_c)


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    doc2word = list(prepro_file_load("doc2word").items())
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())
    D, W = (dictionary.num_docs, len(dictionary))

    train_docs, test_docs = train_test_split(doc2word, test_size=0.33)

    word_topic_assignment, document_topic, document_topic_count, topic_word, topic_word_c = random_initialize(doc2word)
    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, document_topic, document_topic_count, topic_word, topic_word_c,
                       word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ",
              perplexity(test_docs, document_topic, document_topic_count, topic_word, topic_word_c),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, dictionary, num_topics, topic_word))
    model = Model(num_topics, alpha, beta, document_topic, topic_word, "standard")
    model.save_model()
    print(get_topics(dictionary, num_topics, topic_word))
