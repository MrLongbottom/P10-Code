import pickle
from sklearn.model_selection import train_test_split
import time
from typing import List
from scipy import sparse
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
    #s0_topic = np.zeros([s1_num])
    #s1_topic = np.zeros([s1_num, s2_num])
    np.random.seed()
    s2_topic = np.zeros([s2_num, M])
    middle_counts = []
    word_topic_assignment = []
    for d, doc in tqdm(documents):
        sp = sparse.dok_matrix((s1_num, s2_num), np.intc)
        #tax = doc2tax[d]
        currdoc = []
        for w in doc:
            # TODO add option based on observed tax
            # TODO rest maybe just random
            """
            div_1 = np.divide(s0_topic + alpha, len(doc) + (s1_num * alpha))
            # s1_state = [len([x for x in curr_doc if x[0] == s1]) for s1 in range(s1_num)]
            div_2 = np.divide(s1_topic.T + alpha, s0_topic + (s2_num * alpha))
            div_3 = np.divide(s2_topic[:, w] + alpha, s1_topic.sum(axis=0) + (M * alpha))
            pz = np.multiply(np.multiply(div_1, div_2).T, div_3)
            z = np.random.multinomial(1, pz.flatten() / pz.sum()).argmax()
            """
            z = np.random.randint(s1_num*s2_num)
            z1 = math.floor(z / s2_num)
            z2 = z % s2_num
            sp[z1, z2] += 1
            currdoc.append((z1, z2))
            s2_topic[z2, w] += 1
        word_topic_assignment.append(currdoc)
        middle_counts.append(sparse.coo_matrix(sp))
    return word_topic_assignment, middle_counts, s2_topic


def decrease_counts(assignment, middle_counts, s2, word, d):
    middle_counts[d][assignment[0], assignment[1]] -= 1
    s2[assignment[1], word] -= 1


def increase_counts(assignment, middle_counts, s2, word, d):
    middle_counts[d][assignment[0], assignment[1]] += 1
    s2[assignment[1], word] += 1


def gibbs_sampling(documents: List[np.ndarray],
                   wta,
                   middle_counts,
                   s2_topic):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param word_topic_assignment: A list of documents where each index is the given words topic
    :param topic_count: the number of the times each topic is used
    :param documents: a list of documents with their word ids
    :param cat_topic: a matrix describing the number of times each topic within each category
    :param topic_word: a matrix describing the number of times each word within each topic
    """

    # TODO alpha estimations
    # s0_alphas = np.divide(s0_topic, np.sum(s0_topic))
    # s1_alphas = np.divide(s0_topic, np.sum(s0_topic))

    for d_index, doc in documents:
        #tax = doc2tax[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = wta[d_index][w_index]
            decrease_counts(topic, middle_counts, s2, word, d_index)

            # TODO add option based on observed tax
            div_1 = np.divide(middle_counts[d_index].sum(axis=0) + alpha, len(doc) + (s1_num * alpha))
            div_2 = np.divide(s1_topic.T + alpha, s0_topic + (s2_num * alpha))
            div_3 = np.divide(s2_topic[:, word] + alpha, s1_topic.sum(axis=0) + (M * beta))

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
    iterationNum = 10
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

    word_topic_assignment, middle_counts, s2 = random_initialize(doc2word)

    # things needed to calculate coherence
    doc2bow, dictionary, texts = prepro_file_load('doc2bow', folder_name=folder), \
                                 prepro_file_load('corpora', folder_name=folder), \
                                 list(prepro_file_load('doc2pre_text', folder_name=folder).values())

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(doc2word, word_topic_assignment, middle_counts, s2)
        print(time.strftime('%X'), "Iteration: ", i, " Completed")

    print(get_topics(corpora, s2_num, s2))
