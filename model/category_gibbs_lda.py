import pickle
import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utility
from gibbs_utility import decrease_count, increase_count, perplexity, get_coherence, get_topics, mean_topic_diff, \
    cat_perplexity
from preprocess.preprocessing import prepro_file_load, load_memmap_matrix


def random_initialize(documents):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    category_topic = np.zeros([num_categories, num_topics]) + alpha
    topic_word = np.zeros([num_topics, M]) + beta
    topic_c = np.zeros([num_topics]) + M * beta
    wt_assignment = []
    for d, doc in tqdm(list(enumerate(documents))):
        curr_doc = []
        cat = doc2category[d]
        for w in doc:
            pz = np.divide(np.multiply(category_topic[cat, :], topic_word[:, w]), topic_c)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            curr_doc.append(z)
            category_topic[cat, z] += 1
            topic_word[z, w] += 1
            topic_c[z] += 1
        wt_assignment.append(curr_doc)
    return wt_assignment, category_topic, topic_word, topic_c


def gibbs_sampling(documents: List[np.ndarray],
                   cat_topic: np.ndarray,
                   topic_word: np.ndarray,
                   topic_count: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_count: the number of the times each topic is used
    :param documents: a list of documents with their word ids
    :param cat_topic: a matrix describing the number of times each topic within each category
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in enumerate(documents):
        c_index = doc2category[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(topic, topic_word, cat_topic, c_index, word, topic_count)

            # Sample a new topic based on cat_topic and topic word
            # and assign it to the word we are working with
            # np.multiply(np.divide(cat_topic[c_index,:], topic_count)
            pz = np.divide(np.multiply(cat_topic[c_index, :], topic_word[:, word]), topic_count)
            topic = np.random.multinomial(1, pz / pz.sum()).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(topic, topic_word, cat_topic, c_index, word, topic_count)


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    doc2category = prepro_file_load("doc2category")
    num_categories = len(set(doc2category.values()))
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    paths = utility.load_dict_file("../paths.csv")
    shape = (corpora.num_docs, len(corpora))
    doc_word_matrix = load_memmap_matrix('../' + paths["doc_word_matrix"])
    N = doc_word_matrix.shape[0]
    M = doc_word_matrix.shape[1]

    documents = [np.nonzero(x)[0] for x in doc_word_matrix]
    train_docs, test_docs = train_test_split(documents, test_size=0.33, shuffle=True)
    word_topic_assignment, category_topic_dist, topic_word_dist, topic_count = random_initialize(train_docs)

    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, category_topic_dist, topic_word_dist, topic_count, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              " Perplexity: ", cat_perplexity(test_docs, category_topic_dist, topic_word_dist, topic_count),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, corpora, num_topics, topic_word_dist),
              " Topic Diff: ", mean_topic_diff(topic_word_dist))
    print(get_topics(corpora, num_topics, topic_word_dist))