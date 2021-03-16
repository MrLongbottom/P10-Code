import pickle
import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utility
from gibbs_utility import decrease_count, increase_count, perplexity, get_coherence, get_topics
from preprocess.preprocessing import load_memmap_matrix, prepro_file_load


def random_initialize(documents: List[np.ndarray]):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    doc_topic = np.zeros([N, num_topics]) + alpha
    topic_word = np.zeros([num_topics, M]) + beta
    topic_count = np.zeros([num_topics]) + M * beta
    word_topic_assignment = []
    for d, doc in tqdm(list(enumerate(documents))):
        curr_doc = []
        for w in doc:
            pz = np.divide(np.multiply(doc_topic[d, :], topic_word[:, w]), topic_count)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            curr_doc.append(z)
            doc_topic[d, z] += 1
            topic_word[z, w] += 1
            topic_count[z] += 1
        word_topic_assignment.append(curr_doc)
    return word_topic_assignment, doc_topic, topic_word, topic_count


def gibbs_sampling(documents: List[np.ndarray],
                   doc_topic: np.ndarray,
                   topic_word: np.ndarray,
                   topic_count: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_count: the number of the times each topic is used
    :param documents: a list of documents with their word ids
    :param doc_topic: a matrix describing the number of times each topic within each document
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in enumerate(documents):
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_count(topic, topic_word, doc_topic, d_index, word, topic_count)

            # Sample a new topic based on doc_topic and topic word
            # and assign it to the word we are working with
            pz = np.divide(np.multiply(doc_topic[d_index, :], topic_word[:, word]), topic_count)
            topic = np.random.multinomial(1, pz / pz.sum()).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(topic, topic_word, doc_topic, d_index, word, topic_count)


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    paths = utility.load_dict_file("../paths.csv")
    shape = (corpora.num_docs, len(corpora))
    # TODO convert to iterator?
    doc2word = list(prepro_file_load('doc2word', folder_name=None).values())
    N, M = (corpora.num_docs, len(corpora))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, shuffle=True)
    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())

    word_topic_assignment, document_topic_dist, topic_word_dist, topic_count = random_initialize(train_docs)
    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, document_topic_dist, topic_word_dist, topic_count, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ",
              perplexity(test_docs, document_topic_dist, topic_word_dist, topic_count),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, corpora, num_topics, topic_word_dist))
    print(get_topics(corpora, num_topics, topic_word_dist))
