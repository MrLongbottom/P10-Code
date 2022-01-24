import time
from typing import List

import time
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gibbs_utility import get_coherence, mean_topic_diff, get_topics, _conditional_distribution_combination
from model.save import MultiModel
from preprocess.preprocessing import prepro_file_load

# Topic model using gibbs sampling with integrated author and category data
# Each author and category have their own topic distribution
# Works better if each document ALSO has their own topic distribution
# All topic distributions are combined to form a single topic distribution
# or various other integrations that include more speciality for each document while still integrating metadata
# THIS MODEL IS NOT VERY USEFUL!

def random_initialize(documents):
    """
    Randomly initialisation of the word topics
    :param documents: a list of documents with their word ids
    :return: word_topic_assignment,
    """
    print("Random Initilization")
    author_topic = np.zeros([num_authors, num_topics]) + alpha
    author_topic_c = np.zeros([num_authors]) + num_topics * alpha
    category_topic = np.zeros([num_categories, num_topics]) + alpha
    category_topic_c = np.zeros([num_categories]) + num_topics * alpha
    topic_word = np.zeros([num_topics, W]) + beta
    topic_word_c = np.zeros([num_topics]) + W * beta
    wt_assignment = []
    for d_index, doc in tqdm(documents):
        curr_doc = []
        author = doc2author[d_index]
        category = doc2category[d_index]
        for word in doc:
            pz = _conditional_distribution_combination(author, category, word, author_topic, author_topic_c,
                                                       category_topic, category_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            curr_doc.append(topic)
            increase_auth_cat(author, category, word, topic, author_topic, author_topic_c,
                              category_topic, category_topic_c, topic_word, topic_word_c)
        wt_assignment.append(curr_doc)
    return wt_assignment, author_topic, author_topic_c, category_topic, category_topic_c, topic_word, topic_word_c


def increase_auth_cat(author, category, word, topic, author_topic, author_topic_c,
                      category_topic, category_topic_c, topic_word, topic_word_c):
    author_topic[author, topic] += 1
    author_topic_c[author] += 1
    category_topic[category, topic] += 1
    category_topic_c[category] += 1
    topic_word[topic, word] += 1
    topic_word_c[topic] += 1


def decrease_auth_cat(author, category, word, topic, author_topic, author_topic_c,
                      category_topic, category_topic_c, topic_word, topic_word_c):
    author_topic[author, topic] -= 1
    author_topic_c[author] -= 1
    category_topic[category, topic] -= 1
    category_topic_c[category] -= 1
    topic_word[topic, word] -= 1
    topic_word_c[topic] -= 1


def gibbs_sampling(documents: List[np.ndarray],
                   author_topic: np.ndarray,
                   author_topic_c: np.ndarray,
                   category_topic: np.ndarray,
                   category_topic_c: np.ndarray,
                   topic_word: np.ndarray,
                   topic_word_c: np.ndarray,
                   word_topic_assignment: List[List[int]]):
    """
    Takes a set of documents and samples a new topic for each word within each document.
    :param documents: a list of documents with their word ids
    :param author_topic: a matrix describing the number of times each topic is used for each author
    :param author_topic_c: a array of counts of authors
    :param category_topic: a matrix describing the number of times each topic is used for each category
    :param category_topic_c: a array of counts of categories
    :param topic_word: a matrix describing the number of times each word within each topic
    :param topic_word_c: the number of the times each topic is used within words
    :param word_topic_assignment: A list of documents where each index is the given words topic



    """
    for d_index, doc in documents:
        author = doc2author[d_index]
        category = doc2category[d_index]
        for w_index, word in enumerate(doc):
            # Find the topic for the given word a decrease the topic count
            topic = word_topic_assignment[d_index][w_index]
            decrease_auth_cat(author, category, word, topic, author_topic, author_topic_c,
                              category_topic, category_topic_c, topic_word, topic_word_c)

            # Sample a new topic based on author_topic and topic word
            # and assign it to the word we are working with
            pz = _conditional_distribution_combination(author, category, word, author_topic, author_topic_c,
                                                       category_topic, category_topic_c, topic_word, topic_word_c)
            topic = np.random.multinomial(1, pz).argmax()
            word_topic_assignment[d_index][w_index] = topic

            # And increase the topic count
            increase_auth_cat(author, category, word, topic, author_topic, author_topic_c,
                              category_topic, category_topic_c, topic_word, topic_word_c)


def triple_perplexity(documents: List[np.ndarray],
                      author_topic, author_topic_c,
                      category_topic, category_topic_c,
                      topic_word, topic_word_c) -> float:
    """
    Calculates the perplexity based on the documents given
    :param documents: a list of documents with word ids
    :return: the perplexity of the documents given
    """
    n = 0
    ll = 0.0
    for d, doc in documents:
        author = doc2author[d]
        category = doc2category[d]
        for w in doc:
            ll += np.log(((topic_word[:, w] / topic_word_c) *
                          (category_topic[category, :] / category_topic_c[category]) *
                          (author_topic[author, :] / author_topic_c[author])).sum())
            n += 1
    return np.exp(ll / (-n))


if __name__ == '__main__':
    alpha = 0.01
    beta = 0.1
    iterationNum = 50
    num_topics = 90
    doc2author = prepro_file_load("doc2author")
    doc2category = prepro_file_load("doc2category")
    num_authors = len(set(list(doc2author.values())))
    num_categories = len(set(list(doc2category.values())))
    doc2word = list(prepro_file_load("doc2word").items())
    doc2bow, dictionary, texts = prepro_file_load('doc2bow'), prepro_file_load('corpora'), list(
        prepro_file_load('doc2pre_text').values())
    D, W = (dictionary.num_docs, len(dictionary))
    train_docs, test_docs = train_test_split(doc2word, test_size=0.33, random_state=1337)

    word_topic_assignment, author_topic, author_topic_c, category_topic, category_topic_c, topic_word, topic_word_c = \
        random_initialize(doc2word)

    for i in tqdm(range(0, iterationNum)):
        gibbs_sampling(train_docs,
                       author_topic, author_topic_c,
                       category_topic, category_topic_c,
                       topic_word, topic_word_c,
                       word_topic_assignment)
        print(time.strftime('%X'), "Iteration: ", i, " Completed",
              " Perplexity: ",
              triple_perplexity(test_docs,
                                author_topic, author_topic_c,
                                category_topic, category_topic_c,
                                topic_word, topic_word_c),
              " Coherence: ", get_coherence(doc2bow, dictionary, texts, num_topics, topic_word),
              " Topic Diff: ", mean_topic_diff(topic_word))
        model = MultiModel(num_topics, alpha, beta,
                           author_topic, author_topic_c,
                           category_topic, category_topic_c,
                           topic_word, topic_word_c,
                           "author_category")
        model.save_model()
    print(get_topics(dictionary, num_topics, topic_word))
