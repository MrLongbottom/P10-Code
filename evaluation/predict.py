import math
import pickle

import scipy.stats
from tqdm import tqdm
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import dirichlet

from preprocess.preprocessing import prepro_file_load
from statistics import mean

_SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64


def decorate(doc_top, top_words):
    # TODO make it so that smaller values are primarily the ones getting punished rather than the big values.
    # either do modification followed by normalization
    # - normalization will fuck up initial values so that big values become smaller, unless we modify everything
    # or modifications with equal opposite modification elsewhere.
    # - how to ensure that there is 'enough' to take from / give?
    # IDEA: find bad values and remove them to create a 'bank',
    # then increase lower values starting with the best similarity-value combos,
    # until the bank runs dry
    # (it might be better if the bank is a priority with some sort of cutoff to indicate when a transfer is no longer worth it)
    # definition of high similarity?

    # construct topic-similarity matrix
    topic_sim = similarity_matrix(top_words, hellinger_sim)
    # Add add values of other topics based on similarity
    new_doc_top = np.copy(doc_top)
    new_doc_top = new_doc_top * new_doc_top
    for i1, t1 in enumerate(new_doc_top):
        for i2, t2 in enumerate(new_doc_top):
            if i1 != i2:
                sim = topic_sim[i1, i2] if topic_sim[i1, i2] != 0 else topic_sim[i2, i1]
                new_doc_top[i1] += (doc_top[i2] * sim) ** 2
    new_doc_top = new_doc_top / new_doc_top.sum()
    return new_doc_top


def similarity_matrix(things, sim_func):
    sim_matrix = np.zeros(shape=(len(things), len(things)))
    for t1 in range(len(things)):
        for t2 in range(len(things)):
            if t1 != t2: #and sim_matrix[t2, t1] == 0:
                sim_matrix[t1, t2] = sim_func(things[t1], things[t2])
    return sim_matrix


def decorate2(doc_top, top_words):
    topic_sim = similarity_matrix(top_words, hellinger_sim)

    new_doc_top = np.copy(doc_top)
    # make priority list
    # based on best match of topic value multiplied by similarity to that topic
    # highest priority indexes should have their doc-top value increased, while low priority index should be lowered
    priority = {i: max([topic_sim[i, i2] * doc_top[i2] for i2 in range(len(top_words)) if i != i2]) for i, x in enumerate(top_words)}
    # TODO similarities are currently too similar to make any values stand out
    print('test')
    # TODO define cutoff
    # TODO make trades


def hellinger_dis(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger_sim(p, q):
    return 1 - hellinger_dis(p, q)


if __name__ == '__main__':

    folder = "full"
    doc2auth = prepro_file_load("doc2author", folder_name=folder)
    id2auth = prepro_file_load("id2author", folder_name=folder)

    auth2doc = {}
    for doc, auth in doc2auth.items():
        auth2doc[auth] = auth2doc.get(auth, []) + [doc]

    in_folder = "test"
    path = f"../model/generated_files/{in_folder}/"

    with open(path + "wta.pickle", "rb") as file:
        wta = pickle.load(file)
    with open(path + "middle.pickle", "rb") as file:
        middle = pickle.load(file)
    with open(path + "topic_word.pickle", "rb") as file:
        topic_word = pickle.load(file)
    with open(path + "topic_word_dists.pickle", "rb") as file:
        top_word_dists = pickle.load(file)
    with open(path + "document_topic_dists.pickle", "rb") as file:
        doc_top_dists = pickle.load(file)

    """
    # Test case
    doc_top = np.array([0.4, 0.3, 0.10, 0.20])
    top_word = {0: np.array([0.65, 0.0, 0.0, 0.35]),
                1: np.array([0.05, 0.35, 0.15, 0.45]),
                2: np.array([0.55, 0.0, 0.15, 0.3]),
                3: np.array([0, 0.45, 0.35, 0.2])}
    
    decorate(doc_top, top_word)
    """

    # Dirichlet test case
    topic_num = 90
    word_num = 69192
    top_dir = dirichlet(np.full(topic_num, 0.1))
    word_dir = dirichlet(np.full(word_num, 0.01))
    top_sample = top_dir.rvs(size=1)[0]
    word_samples = word_dir.rvs(size=topic_num)
    decorate2(top_sample, word_samples)


    #decorate2(doc_top_dists[2][0], top_word_dists[2])

    # For others Measure distance between topic-distributions for the test document and other documents written by
    # the same author. NOTE even if two topics are using the same words, documents using these two topics separately
    # will be considered completely different.

    # too slow
    # 1000 - 34:30
    doc2auth_predictions = {}
    doc2auth_prediction = {}
    for id, doc_top in tqdm(list(doc_top_dists[2].items())[:100]):
        # compare to all other documents
        doc_sims = np.zeros(len(wta))
        for id2, doc_top2 in doc_top_dists[2].items():
            doc_sims[id2] = hellinger_dis(doc_top, doc_top2)
        # compare to authors (via their docs)
        auth_sims = {}
        for auth, docs in auth2doc.items():
            auth_sims[auth] = np.mean(np.array([doc_sims[x] for x in docs]))
        doc2auth_predictions[id] = auth_sims
        doc2auth_prediction[id] = max(auth_sims, key=auth_sims.get)

    """
    # should be faster if test set is sufficiently large..
    # doc_top_sim = np.zeros((int(len(wta)/2), int(len(wta)/2)))
    # 1000 - 40:28
    doc_top_sim = {}
    doc2auth_predictions2 = {}
    doc2auth_prediction2 = {}
    for id, doc_top in tqdm(list(doc_top_dists.items())[:1000]):
        for id2, doc_top2 in doc_top_dists.items():
            if id != id2 and (id, id2) and (id2, id) not in doc_top_sim:
                doc_top_sim[id, id2] = hellinger1(doc_top, doc_top2)
        auth_sims = {}
        for auth, docs in auth2doc.items():
            auth_sims[auth] = np.mean(np.array(
                [doc_top_sim[(id, x)] for x in docs if (id, x) in doc_top_sim] + [doc_top_sim[(x, id)] for x in docs if
                                                                                  (x, id) in doc_top_sim]))
        doc2auth_predictions2[id] = auth_sims
        doc2auth_prediction2[id] = max(auth_sims, key=auth_sims.get)
    """

    # alternative add-on: decorate all doc-top distributions to increase values on topics that are similar to other topics with high value.
    print('hi')

    # For AT
    # Find the probability of each author having written each document
