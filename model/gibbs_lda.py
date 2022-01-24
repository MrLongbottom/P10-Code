import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import increase_count, decrease_count, perplexity, get_coherence, get_topics, \
    _conditional_distribution, compute_metrics_on_saved_model
from model.save import Model
from preprocess.preprocessing import prepro_file_load
import pickle

# Topic model using gibbs sampling


def random_initialize(documents: List[np.ndarray]):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    doc_topic_distribution = np.zeros([D, num_topics]) + alpha
    doc_topic_count = np.zeros([D]) + num_topics * alpha
    topic_word_distribution = np.zeros([num_topics, W]) + beta
    topic_word_count = np.zeros([num_topics]) + W * beta
    wt_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        for word in doc:
            pz = _conditional_distribution(d_index, word, doc_topic_distribution, doc_topic_count,
                                           topic_word_distribution, topic_word_count)
            topic = np.random.multinomial(1, pz).argmax()
            curr_doc.append(topic)
            increase_count(d_index, word, topic, doc_topic_distribution, doc_topic_count, topic_word_distribution,
                           topic_word_count)
        wt_assignment.append(curr_doc)
    return wt_assignment, doc_topic_distribution, doc_topic_count, topic_word_distribution, topic_word_count


def gibbs_sampling(documents: List[np.ndarray],
                   doc_topic_dist: np.ndarray,
                   doc_topic_count: np.ndarray,
                   topic_word_dist: np.ndarray,
                   topic_word_count: np.ndarray,
                   wt_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param doc_topic_count: 
    :param doc_topic_count:
    :param topic_word_count:
    :param wt_assignment: A list of documents where each index is the given words topic
    :param documents: a list of documents with their word ids
    :param doc_topic_dist: a matrix describing the number of times each topic within each document
    :param topic_word_dist: a matrix describing the number of times each word within each topic
    """
    for d_index, doc in documents:
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = wt_assignment[d_index][w_index]
            decrease_count(d_index, word, topic, doc_topic_dist, doc_topic_count, topic_word_dist, topic_word_count)

            # Sample a new topic based on doc_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution(d_index, word, doc_topic_dist, doc_topic_count, topic_word_dist,
                                           topic_word_count)
            topic = np.random.multinomial(1, pz).argmax()
            wt_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_count(d_index, word, topic, doc_topic_dist, doc_topic_count, topic_word_dist, topic_word_count)


if __name__ == '__main__':
    alpha = 0.01
    beta = 0.1
    iterationNum = 100
    num_topics = 90
    doc2word = list(prepro_file_load("doc2word", folder_name='nice').items())
    doc2bow, dictionary, texts = prepro_file_load('doc2bow', folder_name='nice'), prepro_file_load('corpora', folder_name='nice'), list(
        prepro_file_load('doc2pre_text', folder_name='nice').values())
    D, W = (dictionary.num_docs, len(dictionary))

    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, random_state=1337)

    word_topic_assignment, document_topic, document_topic_count, topic_word, topic_word_c = random_initialize(doc2word)
    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs, document_topic, document_topic_count, topic_word, topic_word_c,
                       word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ",
              perplexity(test_docs, document_topic, document_topic_count, topic_word, topic_word_c),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, num_topics, topic_word))
    model = Model(num_topics, alpha, beta, document_topic, document_topic_count, topic_word, topic_word_c, "standard")
    model.save_model()
    print(get_topics(dictionary, num_topics, topic_word))

    # doc2word = list(prepro_file_load("doc2word").items())
    # train_docs, test_docs = train_test_split(doc2word, test_size=0.33, random_state=1337)
    # print(compute_metrics_on_saved_model("90_0.01_0.1_author_category_MultiModel", test_docs, True))

    # calculate document-topic distribution
    doc_top_dists = {}
    for id, doc_info in tqdm(enumerate(word_topic_assignment)):
        doc_dist = np.zeros(shape=num_topics)
        for word in doc_info:
            doc_dist[word] += 1
        doc_top_dists[id] = doc_dist / doc_dist.sum()

    # calculate topic-word distribution
    top_word_dists = {i: np.zeros(W) for i in range(num_topics)}
    for id, doc_info in tqdm(enumerate(word_topic_assignment)):
        for word_pos, word_topic in enumerate(doc_info):
            word_id = doc2word[id][1][word_pos]
            top_word_dists[word_topic][word_id] += 1
    for t in range(num_topics):
        top_word_dists[t] = top_word_dists[t] / top_word_dists[t].sum()

    out_folder = 'lda'
    path = f"generated_files/{out_folder}/"

    with open(path + "wta.pickle", "wb") as file:
        pickle.dump(word_topic_assignment, file)
    with open(path + "topic_word.pickle", "wb") as file:
        pickle.dump(topic_word, file)
    with open(path + "topic_to_word.pickle", "wb") as file:
        pickle.dump(topic_word_c, file)
    with open(path + "topic_word_dists.pickle", "wb") as file:
        pickle.dump(top_word_dists, file)
    with open(path + "document_topic_dists.pickle", "wb") as file:
        pickle.dump(doc_top_dists, file)
