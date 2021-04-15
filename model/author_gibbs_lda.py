import pickle
import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import get_coherence, mean_topic_diff, get_topics, decrease_count, increase_count, \
    _conditional_distribution, x_perplexity
from model.save import Model
from preprocess.preprocessing import prepro_file_load


def random_initialize(documents):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    author_topic = np.zeros([num_authors, num_topics]) + alpha
    topic_word = np.zeros([num_topics, W]) + beta
    word_topic_c = np.zeros([num_topics]) + W * beta
    author_topic_c = np.zeros([num_authors]) + num_topics * alpha
    wt_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        author = doc2author[d_index]
        for word in doc:
            pz = _conditional_distribution(author, word, topic_word, author_topic, word_topic_c, author_topic_c)
            topic = np.random.multinomial(1, pz).argmax()
            curr_doc.append(topic)
            author_topic[author, topic] += 1
            topic_word[topic, word] += 1
            word_topic_c[topic] += 1
            author_topic_c[author] += 1
        wt_assignment.append(curr_doc)
    return wt_assignment, author_topic, topic_word, word_topic_c, author_topic_c


def gibbs_sampling(documents: List[np.ndarray],
                   author_topic: np.ndarray,
                   topic_word: np.ndarray,
                   word_topic_count: np.ndarray,
                   doc_topic_count: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param doc_topic_count: the number of the times each topic is used within docs
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param word_topic_count: the number of the times each topic is used within words
    :param documents: a list of documents with their word ids
    :param author_topic: a matrix describing the number of times each topic is used for each author
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        a_index = doc2author[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(topic, topic_word, author_topic, a_index, word, word_topic_count, doc_topic_count)

            # Sample a new topic based on author_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution(a_index, word, topic_word, author_topic, word_topic_count, doc_topic_count)
            topic = np.random.multinomial(1, pz).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(topic, topic_word, author_topic, a_index, word, word_topic_count, doc_topic_count)


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    doc2author = prepro_file_load("doc2author")
    num_authors = len(set(list(doc2author.values())))
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    doc2word = list(prepro_file_load("doc2word").items())
    D, W = (corpora.num_docs, len(corpora))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, shuffle=True)

    word_topic_assignment, author_topic_dist, topic_word_dist, wt_count, dt_count = random_initialize(doc2word)

    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, author_topic_dist, topic_word_dist, wt_count, dt_count, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              " Perplexity: ",
              x_perplexity(test_docs, author_topic_dist, topic_word_dist, wt_count, dt_count, doc2author),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, corpora, num_topics, topic_word_dist),
              " Topic Diff: ", mean_topic_diff(topic_word_dist))
    model = Model(num_topics, alpha, beta, author_topic_dist, topic_word_dist, "author")
    model.save_model()
    print(get_topics(corpora, num_topics, topic_word_dist))
