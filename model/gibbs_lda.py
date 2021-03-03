import pickle
import time
from functools import partial
from multiprocessing import Pool, Value
import multiprocessing
import numpy as np
from tqdm import tqdm

num_thread = 4


def gibbs(documents):
    vocab_ranges = [x for x in range(0, W, int(W / num_thread))]
    if len(vocab_ranges) != num_thread+1:
        vocab_ranges.append(W-1)
    for x in range(num_thread):
        vocab_thread_assignment = [(vocab_ranges[(index + x) % num_thread],
                                    vocab_ranges[((index + x) % num_thread) + 1])
                                   for index in range(num_thread)]
        doc_vocab_pairs = zip(documents, vocab_thread_assignment)
        print('starting pool')
        with Pool(processes=num_thread) as p:
            r = p.map(partial(sampling, Z, [D, W, K, alpha, beta]), doc_vocab_pairs)
            print(r)
        print()


def sampling(Z, consts, doc_vocab):
    D, W, K, alpha, beta = consts
    documents, vocab_range = doc_vocab

    # Make counts
    document_topic_dist = np.zeros([D, K]) + alpha
    topic_word_dist = np.zeros([K, W]) + beta
    nz = np.zeros([K])
    for i, d in enumerate(Z):
        for j, z in enumerate(d):
            document_topic_dist[i, z] += 1
            topic_word_dist[z, j] += 1
            nz[z] += 1

    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            if vocab_range[0] < word < vocab_range[1]:
                # Forget current
                topic = Z[d_index][w_index]
                document_topic_dist[d_index, topic] -= 1
                topic_word_dist[topic, w_index] -= 1
                nz[topic] -= 1

                # Sample a new topic and assign it to the topic assignment matrix
                pz = np.divide(np.multiply(document_topic_dist[d_index, :], topic_word_dist[:, word]), nz)
                topic = np.random.multinomial(1, pz / pz.sum()).argmax()
                Z[d_index][w_index] = topic

                # Increase the counts by 1
                document_topic_dist[d_index, topic] += 1
                topic_word_dist[topic, w_index] += 1
                nz[topic] += 1
    return Z, document_topic_dist, topic_word_dist, nz


def increase(topic, doc_topic, topic_word, word, d_index):
    doc_topic[d_index, topic] += 1
    topic_word[topic, word] += 1
    nz[topic] += 1


def decrease(topic, doc_topic, topic_word, word, d_index):
    doc_topic[d_index, topic] -= 1
    topic_word[topic, word] -= 1
    nz[topic] -= 1


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
