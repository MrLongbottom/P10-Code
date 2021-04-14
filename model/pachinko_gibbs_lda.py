import pickle
from sklearn.model_selection import train_test_split
import time
from typing import List
from scipy import sparse
import numpy as np
from tqdm import tqdm
import random

import utility
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
    np.random.seed()
    middle_layers = [[] for _ in range(N)]
    topic_to_word = np.zeros([layer_lengths[len(layer_lengths)-1], M])
    word_topic_assignment = []
    for d, doc in tqdm(documents):
        mid_doc_layers = [np.zeros((layer_lengths[x], layer_lengths[x+1]), dtype=np.intc) for x in range(len(layer_lengths)-1)]
        tax_ids = doc2tax[d]
        tax = tax2topic_id(tax_ids)
        currdoc = []
        for w in doc:
            # randomly choose a taxonomy chain if there are multiple
            if len(tax) > 1:
                word_tax = [x for x in tax[random.randrange(len(tax))]]
            elif len(tax) == 1:
                word_tax = [x for x in tax[0]]
            else:
                word_tax = []

            while len(word_tax) < len(layer_lengths):
                word_tax.append(np.random.randint(layer_lengths[len(word_tax)]))
            z = tuple(word_tax)
            # apply various counts
            for i, matrix in enumerate(mid_doc_layers):
                matrix[z[i], z[i+1]] += 1
            currdoc.append(z)
            topic_to_word[z[len(z)-1], w] += 1
        word_topic_assignment.append(currdoc)
        middle_layers[d] = mid_doc_layers
    return word_topic_assignment, middle_layers, topic_to_word


def decrease_counts(topics, middle_layers, middle_sums, topic_to_word, topic_to_word_sums, word, d):
    for i, x in enumerate(middle_layers[d]):
        x[topics[i], topics[i+1]] -= 1
    for i, x in enumerate(middle_sums):
        x[topics[i]] -= 1
    topic_to_word[topics[len(topics)-1], word] -= 1
    topic_to_word_sums[topics[len(topics)-1]] -= 1


def increase_counts(topics, middle_layers, middle_sums, topic_to_word, topic_to_word_sums, word, d):
    for i, x in enumerate(middle_layers[d]):
        x[topics[i], topics[i + 1]] += 1
    for i, x in enumerate(middle_sums):
        x[topics[i]] += 1
    topic_to_word[topics[len(topics) - 1], word] += 1
    topic_to_word_sums[topics[len(topics) - 1]] += 1


def gibbs_sampling(documents: List[np.ndarray],
                   wta,
                   middle_layers,
                   topic_to_word):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param middle_layers:
    :param wta: Word-Topic-Assignments. A list of documents where each index is the given words topic
    :param documents: a list of documents with their word ids
    :param topic_to_word: a matrix describing the number of times each word within each topic
    """

    # TODO alpha estimations (might not be needed?)
    # sum calculated to be used later
    topic_to_word_sums = topic_to_word.sum(axis=1)
    letters = ['a', 'b', 'c', 'd']
    letters = {layer_lengths[i]: letters[i] for i in range(len(layer_lengths))}

    for d_index, doc in tqdm(documents):
        # find existing taxonomy
        tax_ids = doc2tax[d_index]
        tax = tax2topic_id(tax_ids)

        # if already full, skip document, as it will just stay in that taxonomy
        if len(tax) == 1 and len(tax[0]) == len(layer_lengths):
            continue

        # sums calculated to be used later
        middle_sums = [x.sum(axis=1) for x in middle_layers[d_index]]

        for w_index, word in enumerate(doc):
            # randomly choose a taxonomy chain if there are multiple
            if len(tax) > 1:
                word_tax = tax[random.randrange(len(tax))]
            elif len(tax) == 1:
                word_tax = tax[0]
            else:
                word_tax = []

            # Find the topic for the given word a decrease the topic count
            topic = wta[d_index][w_index]
            decrease_counts(topic, middle_layers, middle_sums, topic_to_word, topic_to_word_sums, word, d_index)

            # Pachinko Equation (pachinko paper, bottom of page four, extended to three layers)
            divs = []
            if len(word_tax) == 0:
                divs.append(np.divide((middle_sums[0] + alpha), len(doc) + (layer_lengths[0] * alpha)))
            for i in range(len(middle_sums)):
                if len(word_tax) < i+2:
                    divs.append(np.divide((middle_layers[d_index][i] + alpha).T, (middle_sums[i] + (layer_lengths[i+1] * alpha))).T)
            if len(word_tax) < len(layer_lengths):
                divs.append(np.divide((topic_to_word[:, word] + beta), topic_to_word_sums + (M * beta)))

            # TODO make work for any number of dimensions (currently only working with 3 middle layers)
            # Multiply divs together (skip any where taxonomy is already known)
            steps = []
            # if some of the taxonomy is already known
            if len(word_tax) != 0:
                for step in range(len(layer_lengths) - len(word_tax) + 1):
                    if step == 0:
                        if len(divs[len(word_tax)-1].shape) == 2:
                            steps.append(divs[len(word_tax)-1][word_tax[len(word_tax)-1], :])
                        elif len(divs[len(word_tax)-1].shape) == 1:
                            steps.append(divs[len(word_tax)-1][word_tax[len(word_tax)-1]])
                        else:
                            print("fuck")
                        continue
                    else:
                        start = steps[step-1]
                    end = divs[step]
                    letters1 = [letters[x] for x in start.shape]
                    letters2 = [letters[x] for x in end.shape]
                    letters3 = [x for x in letters1]
                    letters3.extend(letters2)
                    letters3 = list(set(letters3))
                    steps.append(np.einsum(''.join(letters1) + ',' + ''.join(letters2) + '->' + ''.join(letters3), start, end))
            # if taxonomy is not known
            else:
                for step in range(len(layer_lengths)):
                    if step == 0:
                        start = divs[0]
                    else:
                        start = steps[step-1]
                    end = divs[step+1]
                    letters1 = [letters[x] for x in start.shape]
                    letters2 = [letters[x] for x in end.shape]
                    letters3 = [x for x in letters1]
                    letters3.extend(letters2)
                    letters3 = list(set(letters3))
                    steps.append(np.einsum(''.join(letters1) + ',' + ''.join(letters2) + '->' + ''.join(letters3), start, end))

            # convert matrix into flat array to sample a word_taxonomy combination
            flat = np.asarray(steps[len(steps)-1].flatten() / steps[len(steps)-1].sum())
            z = np.random.multinomial(1, flat)

            # reshape to find the chosen word_taxonomy combination
            z = z.reshape(layer_lengths[len(word_tax):])
            topic = [x for x in word_tax]
            topic.extend([x[0] for x in np.where(z == z.max())])
            topic = tuple(topic)

            # Assign and increase topic count
            word_topic_assignment[d_index][w_index] = topic
            increase_counts(topic, middle_layers, middle_sums, topic_to_word, topic_to_word_sums, word, d_index)


def tax2topic_id(tax_id_list):
    """
    Convert taxonomy id list into id's in taxonomy tree structure
    :param tax_id_list: list of taxonomy id's
    :return: list of taxonomy id's in the taxonomy tree structure (aka. topic id's)
    """
    topic_ids = []
    curr_list = -1
    for tax in tax_id_list:
        tax_name = id2tax[tax]
        if tax_name == '':
            return topic_ids
        elif tax_name in struct_root[0]:
            topic_ids.append([struct_root[0].index(tax_name)])
            curr_list += 1
        elif tax_name in struct_root[len(topic_ids)] and curr_list != -1 and len(topic_ids[curr_list]) < mid_layers_num:
            topic_ids[curr_list].append(struct_root[len(topic_ids)].index(tax_name))
    topic_ids = [x[:mid_layers_num] for x in topic_ids]
    return topic_ids


def taxonomy_structure(layers):
    root = {}
    for d2t in doc2tax.values():
        parent = None
        for t in d2t:
            if id2tax[t] == '':
                continue
            elif parent is None or id2tax[t] == 'EMNER' or id2tax[t] == 'STEDER' or id2tax[t] == 'TEMA' \
                    or id2tax[t] == 'IMPORT' or id2tax[t] == 'INDHOLDSTYPER':
                if id2tax[t] not in root:
                    root[id2tax[t]] = {}
                parent = root[id2tax[t]]
            else:
                if id2tax[t] not in parent:
                    parent[id2tax[t]] = {}
                parent = parent[id2tax[t]]
    struct_root = []
    for l in range(layers):
        if l == 0:
            struct_root.append([x for x in root.items()])
        else:
            struct_root.append([y for x in struct_root[l-1] for y in x[1].items()])
    struct_root = [[y[0] for y in x] for x in struct_root]
    return root, struct_root


if __name__ == '__main__':
    random.seed()
    folder = 'full'
    alpha = 0.1
    beta = 0.1
    iterationNum = 10
    # number of "empty" topics in bottom layer
    # if 'None' no bottom layer of empty topic will be added
    K = 90
    # number of layers to take from taxonomy tree
    mid_layers_num = 2

    doc2tax = prepro_file_load("doc2taxonomy", folder_name=folder)
    id2tax = prepro_file_load("id2taxonomy", folder_name=folder)
    corpora = prepro_file_load("corpora", folder_name=folder)
    doc2word = list(prepro_file_load("doc2word", folder_name=folder).items())
    # number of docs and words
    N, M = (corpora.num_docs, len(corpora))

    # taxonomy tree structure
    root, struct_root = taxonomy_structure(mid_layers_num)
    # tree layer sizes
    layer_lengths = [len(x) for x in struct_root]
    if K is not None:
        layer_lengths.append(K)

    word_topic_assignment, middle_layers, topic_to_word = random_initialize(doc2word)

    # things needed to calculate coherence
    doc2bow, texts = prepro_file_load('doc2bow', folder_name=folder), \
                     list(prepro_file_load('doc2pre_text', folder_name=folder).values())

    print("Starting Gibbs")
    for i in range(0, iterationNum):
        gibbs_sampling(doc2word, word_topic_assignment, middle_layers, topic_to_word)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              "Coherence: ", get_coherence(doc2bow, texts, corpora, layer_lengths[2], topic_to_word))

    topic_words = get_topics(corpora, layer_lengths[2], topic_to_word)
    if K is None:
        topic_words = {struct_root[mid_layers_num-1][i]: topic_words[i] for i in range(len(topic_words))}
    print(topic_words)
