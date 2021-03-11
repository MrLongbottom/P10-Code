import pickle
import time
from typing import List

import numpy as np
from tqdm import tqdm

from preprocess.preprocessing import prepro_file_load
import coherence

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
    for d, doc in tqdm(enumerate(documents)):
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


def increase_count(topic, topic_word, doc_topic, d_index, word, t_count):
    doc_topic[d_index, topic] += 1
    topic_word[topic, word] += 1
    t_count[topic] += 1


def decrease_count(topic, topic_word, doc_topic, d_index, word, t_count):
    doc_topic[d_index, topic] -= 1
    topic_word[topic, word] -= 1
    t_count[topic] -= 1


def perplexity(documents: List[np.ndarray]) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    nd = np.sum(document_topic_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(documents):
        for w in doc:
            ll = ll + np.log(((topic_word_dist[:, w] / topic_count) * (document_topic_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def get_topics(num_of_word_per_topic: int = 10):
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


def get_coherence(doc2bow, dictionary, texts):
    return coherence.coherence(topics=get_topics(), doc2bow=doc2bow, dictionary=dictionary, texts=texts)


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    num_topics = 10
    with open("../preprocess/generated_files/doc_word_matrix.pickle", 'rb') as file:
        doc_word_matrix = pickle.load(file)
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    doc_word_matrix = np.array(doc_word_matrix.to_dense(), dtype=int)
    N = doc_word_matrix.shape[0]
    M = doc_word_matrix.shape[1]
    doc_word_matrix = [np.nonzero(x)[0] for x in doc_word_matrix]

    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('id2pre_text').values())

    word_topic_assignment, document_topic_dist, topic_word_dist, topic_count = random_initialize(doc_word_matrix)
    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(doc_word_matrix, document_topic_dist, topic_word_dist, topic_count, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(doc_word_matrix), " Coherence: ", get_coherence(doc2bow, dictionary, texts))
    print(get_topics(10))
