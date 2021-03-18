import pickle
from sklearn.model_selection import train_test_split
import time
from typing import List

import numpy as np
from tqdm import tqdm

from gibbs_utility import perplexity, get_coherence, mean_topic_diff, get_topics, decrease_count, increase_count, \
    cat_perplexity
from preprocess.preprocessing import prepro_file_load
import math


def random_initialize(documents):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    s0_topic = np.zeros([s1_num])
    s1_topic = np.zeros([s1_num, s2_num])
    s2_topic = np.zeros([s2_num, M])
    wt_assignment = []
    for d, doc in tqdm(documents):
        np.random.seed()
        curr_doc = []
        tax = doc2tax[d]
        for w in doc:
            # TODO add option based on observed tax
            # TODO for training: change 0's to be number of times topic has been chosen in d
            div_1 = np.divide(s0_topic + alpha, len(doc) + (s1_num*alpha))
            # s1_state = [len([x for x in curr_doc if x[0] == s1]) for s1 in range(s1_num)]
            div_2 = np.divide(s1_topic.T + alpha, s0_topic + (s2_num*alpha))
            div_3 = np.divide(s2_topic[:,w] + alpha, s1_topic.sum(axis=0) + (M*alpha))

            pz = np.multiply(np.multiply(div_1, div_2).T, div_3)

            z = np.random.multinomial(1, pz.flatten() / pz.sum()).argmax()
            z1 = math.floor(z / 54)
            z2 = z % 54
            curr_doc.append((z1,z2))
            s0_topic[z1] += 1
            s1_topic[z1, z2] += 1
            s2_topic[z2, w] += 1
        wt_assignment.append(curr_doc)
    return wt_assignment, s0_topic, s1_topic, s2_topic


def decrease_counts(assignment, s0, s1, s2, word):
    s0[assignment[0]] -= 1
    s1[assignment[0], assignment[1]] -= 1
    s2[assignment[1], word] -= 1


def increase_counts(assignment, s0, s1, s2, word):
    s0[assignment[0]] += 1
    s1[assignment[0], assignment[1]] += 1
    s2[assignment[1], word] += 1


def gibbs_sampling(documents: List[np.ndarray],
                   s0_topic: np.ndarray,
                   s1_topic: np.ndarray,
                   s2_topic: np.ndarray,
                   word_topic_assignment):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_count: the number of the times each topic is used
    :param documents: a list of documents with their word ids
    :param cat_topic: a matrix describing the number of times each topic within each category
    :param topic_word: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        tax = doc2tax[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_counts(topic, s0, s1, s2, word)

            # TODO add option based on observed tax
            div_1 = np.divide(s0_topic + alpha, len(doc) + (s1_num * alpha))
            # s1_state = [len([x for x in curr_doc if x[0] == s1]) for s1 in range(s1_num)]
            div_2 = np.divide(s1_topic.T + alpha, s0_topic + (s2_num * alpha))
            div_3 = np.divide(s2_topic[:, word] + alpha, s1_topic.sum(axis=0) + (M * alpha))

            pz = np.multiply(np.multiply(div_1, div_2).T, div_3)

            z = np.random.multinomial(1, pz.flatten() / pz.sum()).argmax()
            topic = (math.floor(z / 54), z % 54)
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_counts(topic, s0, s1, s2, word)


if __name__ == '__main__':
    folder = '2017'
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    doc2tax = prepro_file_load("doc2taxonomy", folder_name=folder)
    id2tax = prepro_file_load("id2taxonomy", folder_name=folder)
    root = {}
    for d2t in doc2tax.values():
        parent = None
        for t in d2t:
            if id2tax[t] == '':
                continue
            elif parent is None or id2tax[t] == '' or id2tax[t] == 'EMNER' or id2tax[t] == 'STEDER':
                if id2tax[t] not in root:
                    root[id2tax[t]] = {}
                parent = root[id2tax[t]]
            else:
                if id2tax[t] not in parent:
                    parent[id2tax[t]] = {}
                parent = parent[id2tax[t]]
    s1_num = len(root)
    s2_num = np.sum([len(x) for x in root.values()])

    corpora = prepro_file_load("corpora", folder_name=folder)
    doc2word = list(prepro_file_load("doc2word", folder_name=folder).items())
    N, M = (corpora.num_docs, len(corpora))

    word_topic_assignment, s0, s1, s2 = random_initialize(doc2word)

    # things needed to calculate coherencecategory_gibbs_lda.py
    doc2bow, dictionary, texts = prepro_file_load('doc2bow', folder_name=folder), \
                                 prepro_file_load('corpora', folder_name=folder), \
                                 list(prepro_file_load('doc2pre_text', folder_name=folder).values())

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(doc2word, s0, s1, s2, word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed")

    print(get_topics(corpora, s2_num, s2))