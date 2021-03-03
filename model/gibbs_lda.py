import pickle
import time
from functools import partial
from multiprocessing import Pool, Value
import multiprocessing
import numpy as np
from tqdm import tqdm

num_thread = 8

gZ = None
gDT = None
gTW = None


def init_globals(gz, gdt, gtw):
    global gZ, gDT, gTW
    gZ = gz
    gDT = gdt
    gTW = gtw


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
            result = p.map(partial(sampling, topic_count), doc_vocab_pairs)
            print()
        print()


def sampling(topic_count, doc_vocab):
    global gZ, gDT, gTW
    documents, vocab_range = doc_vocab
    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            if vocab_range[0] < word < vocab_range[1]:
                # Find the topic for the given word and decrease the counts by 1
                topic = gZ[d_index][w_index].value
                with gDT[d_index][topic].get_lock():
                    gDT[d_index][topic].value -= 1
                with gTW[w_index][topic].get_lock():
                    gTW[w_index][topic].value -= 1
                topic_count[topic] -= 1

                # Sample a new topic and assign it to the topic assignment matrix
                pz = np.divide(np.multiply(gDT[d_index, :], gTW[:, word]), nz)
                topic = np.random.multinomial(1, pz / pz.sum()).argmax()
                with gZ[d_index][w_index].getlock():
                    gZ[d_index][w_index] = topic

                # Increase the counts by 1
                with gDT[d_index][topic].get_lock():
                    gDT[d_index][topic].value += 1
                with gTW[w_index][topic].get_lock():
                    gTW[w_index][topic].value += 1
                topic_count[topic] += 1
    return topic_count


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

    gZ = [[Value('i', i) for i in d] for d in Z]
    gDT = [[Value('i', int(doc_topic[d, k])) for k in range(K)] for d in range(D)]
    gTW = [[Value('i', int(topic_word[k, w])) for w in range(W)] for k in range(K)]
    topic_count = [int(sum(x)) for x in topic_word]

    vocab_ranges = [x for x in range(0, W, int(W / num_thread))]
    if len(vocab_ranges) != num_thread+1:
        vocab_ranges.append(W - 1)

    for x in tqdm(range(num_thread)):
        vocab_thread_assignment = [(vocab_ranges[(index + x) % num_thread],
                                    vocab_ranges[((index + x) % num_thread) + 1])
                                   for index in range(num_thread)]
        doc_vocab_pairs = zip(doc_sets, vocab_thread_assignment)
        print('starting pool')
        with Pool(processes=num_thread, initializer=init_globals, initargs=[gZ, gDT, gTW]) as p:
            r = p.map_async(partial(sampling, topic_count), doc_vocab_pairs, chunksize=1)
            r.wait()
            r1 = r.get()
            print(r1)

    '''
    for i in tqdm(range(0, iterationNum)):
        gibbs(sets)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity(doc_word_matrix))
    '''
