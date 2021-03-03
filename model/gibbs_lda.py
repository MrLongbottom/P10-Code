import pickle
import time
from functools import partial
from multiprocessing import Pool, Value
import multiprocessing
import numpy as np
from tqdm import tqdm
import collections

num_thread = 8


def gibbs(documents):
    vocab_ranges = [x for x in range(0, W, int(W / num_thread))]
    if len(vocab_ranges) != num_thread+1:
        vocab_ranges.append(W-1)
    for x in range(num_thread):
        vocab_thread_assignment = [(vocab_ranges[(index + x) % num_thread],
                                    vocab_ranges[((index + x) % num_thread) + 1])
                                   for index in range(num_thread)]
        doc_vocab_pairs = zip(documents, vocab_thread_assignment)
        with Pool(processes=num_thread) as p:
            r = p.map(partial(sampling, Z, [D, W, K, alpha, beta]), doc_vocab_pairs)
            print([len(x) for x in r])
            keys = [k for d in r for k in d.keys()]
            overlap = [item for item, count in collections.Counter(keys).items() if count > 1]
            if len(overlap) > 0:
                raise Exception('We have thread overlap!')
            combined = {k: v for d in r for k, v in d.items()}
            # update / synchronize Z
            for k, v in combined.items():
                Z[k[0]][k[1]] = v


def compute_counts(Z, consts):
    D, W, K, alpha, beta = consts
    document_topic_dist = np.zeros([D, K]) + alpha
    topic_word_dist = np.zeros([K, W]) + beta
    topic_counts = np.zeros([K])
    for i, d in enumerate(Z):
        for j, z in enumerate(d):
            document_topic_dist[i, z] += 1
            topic_word_dist[z, j] += 1
            topic_counts[z] += 1
    return document_topic_dist, topic_word_dist, topic_counts


def sampling(Z, consts, doc_vocab):
    documents, vocab_range = doc_vocab
    z_change = {}
    # Make counts
    document_topic_dist, topic_word_dist, topic_counts = compute_counts(Z, consts)

    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            if vocab_range[0] < word < vocab_range[1]:
                # Forget current
                topic = Z[d_index][w_index]
                document_topic_dist[d_index, topic] -= 1
                topic_word_dist[topic, w_index] -= 1
                topic_counts[topic] -= 1

                # Sample a new topic and assign it to the topic assignment matrix
                pz = np.divide(np.multiply(document_topic_dist[d_index, :], topic_word_dist[:, word]), topic_counts)
                topic = np.random.multinomial(1, pz / pz.sum()).argmax()
                Z[d_index][w_index] = topic

                # Increase the counts by 1
                document_topic_dist[d_index, topic] += 1
                topic_word_dist[topic, w_index] += 1
                topic_counts[topic] += 1
                z_change[(d_index, w_index)] = topic
    return z_change


def random_initialize(documents, doc_topic, topic_word):
    print("Random Initilization")
    Z = []
    for d, doc in tqdm(enumerate(documents)):
        zCurrentDoc = []
        for w in doc:
            pz = np.divide(np.multiply(doc_topic[d, :], topic_word[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            doc_topic[d, z] += 1
            topic_word[z, w] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)
    return Z, doc_topic - alpha, topic_word - beta


def perplexity(data):
    document_topic_dist, topic_word_dist, topic_counts = compute_counts(Z, [D,W,K,alpha,beta])
    nd = np.sum(document_topic_dist, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(data):
        for w in doc:
            ll = ll + np.log(((topic_word_dist[:, w] / nz) * (document_topic_dist[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    iterationNum = 50
    K = 10
    with open("../preprocess/generated_files/doc_word_matrix.pickle", 'rb') as file:
        doc_word_matrix = pickle.load(file)
    with open("../preprocess/generated_files/corpora", 'rb') as file:
        corpora = pickle.load(file)
    doc_word_matrix = np.array(doc_word_matrix.to_dense(), dtype=int)
    D = doc_word_matrix.shape[0]
    W = doc_word_matrix.shape[1]
    doc_word_matrix = [np.nonzero(x)[0] for x in doc_word_matrix]
    doc_sets = [list(x) for x in chunks(list(enumerate(doc_word_matrix)), int(len(doc_word_matrix) / num_thread))]

    document_topic_dist = np.zeros([D, K]) + alpha
    topic_word_dist = np.zeros([K, W]) + beta
    nz = np.zeros([K]) + W * beta

    Z, doc_topic, topic_word = random_initialize(doc_word_matrix, document_topic_dist, topic_word_dist)
    topic_count = [int(sum(x)) for x in topic_word]

    for i in tqdm(range(0, iterationNum)):
        gibbs(doc_sets)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(doc_word_matrix))
