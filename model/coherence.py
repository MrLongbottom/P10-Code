import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial import distance


# supported measures = {'u_mass', 'c_v', 'c_uci', 'c_npmi'}
def coherence(topics, doc2bow, dictionary, texts, coherence_measure = 'c_v'):
    cm = CoherenceModel(topics=topics, corpus=doc2bow, dictionary=dictionary, texts=texts, coherence=coherence_measure)
    return cm.get_coherence()


def topic_diff(topic_word_dist):
    topic_difference_matrix = np.zeros((topic_word_dist.shape[0]-1, topic_word_dist.shape[0]-1))
    for topics in np.ndindex(topic_difference_matrix.shape):
        # don't compare topics to themselves
        if topics[0] == topics[1]:
            continue
        # fill out distance matrix with distance between topic1 and topic2
        topic_difference_matrix[topics] = distance.jensenshannon(topic_word_dist[topics[0]], topic_word_dist[topics[1]])
    return topic_difference_matrix


def mean_topic_diff(topic_word_dist):
    td = topic_diff(topic_word_dist)
    return np.mean(td)
