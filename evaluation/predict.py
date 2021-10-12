import math
import pickle
from tqdm import tqdm
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

from preprocess.preprocessing import prepro_file_load
from statistics import mean

_SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64


def decorate(doc_top, top_words):
    # TODO make it so that smaller values are primarily the ones getting punished rather than the big values.
    # construct topic-similarity matrix
    topic_sim = np.zeros(shape=(len(top_words), len(top_words)))
    for t1 in tqdm(range(len(top_words))):
        for t2 in range(len(top_words)):
            if t1 != t2 and topic_sim[t2, t1] == 0:
                topic_sim[t1, t2] = hellinger_sim(top_words[t1], top_words[t2])
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


def hellinger_dis(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger_sim(p, q):
    return 1-hellinger_dis(p, q)


if __name__ == '__main__':

    folder = "full"
    doc2auth = prepro_file_load("doc2author", folder_name=folder)
    id2auth = prepro_file_load("id2author", folder_name=folder)

    auth2doc = {}
    for doc, auth in doc2auth.items():
        auth2doc[auth] = auth2doc.get(auth, []) + [doc]

    in_folder = "test"
    path = f"../model/generated_files/{in_folder}/"

    with open(path+"wta.pickle", "rb") as file:
        wta = pickle.load(file)
    with open(path+"middle.pickle", "rb") as file:
        middle = pickle.load(file)
    with open(path+"topic_word.pickle", "rb") as file:
        topic_word = pickle.load(file)
    with open(path+"topic_word_dists.pickle", "rb") as file:
        top_word_dists = pickle.load(file)
    with open(path+"document_topic_dists.pickle", "rb") as file:
        doc_top_dists = pickle.load(file)

    doc_top = np.array([0.4, 0.3, 0.10, 0.20])
    top_word = {0: np.array([0.65, 0.0, 0.0, 0.35]),
                1: np.array([0.05, 0.35, 0.15, 0.45]),
                2: np.array([0.55, 0.0, 0.15, 0.3]),
                3: np.array([0, 0.45, 0.35, 0.2])}

    #decorate(doc_top_dists[2][0], top_word_dists[2])
    decorate(doc_top, top_word)

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
