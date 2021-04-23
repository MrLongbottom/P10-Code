import random
import re
from typing import List

import itertools
import preprocess.preprocessing as pre
from model.save import load_model
from gibbs_utility import get_topics


def color_words(words: List[str], text: str):
    for word in words:
        text = text.replace(" " + word + " ", " \\" + "colorbox{green}" + "{" + word + "} ")
    return text


if __name__ == '__main__':
    corpora = pre.prepro_file_load("corpora")
    doc2pre = pre.prepro_file_load('doc2pre_text')
    doc2raw = pre.prepro_file_load('doc2raw_text')
    model = load_model("../model/90_0.01_0.1_author")
    num_topics = model.num_topics
    topic_word_dist = model.topic_word

    docs = dict(enumerate(model.doc_topic))
    docs = [(k, v) for k, v in docs.items()]  # convert to list of (ID, topics)
    # sample_docs = random.sample(docs, 5)  # sample 5 random documents
    sample_docs = [docs[38668]]
    # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
    sample_docs = [(doc[0], [(index, topic) for index, topic in enumerate(doc[1])]) for doc in sample_docs]

    # sort topics in sampled docs and keep top 3 topics
    sorted_doc_topic = {}
    for doc in sample_docs:
        sorted_doc_topic[doc[0]] = sorted(doc[1], key=lambda topic: topic[1], reverse=True)
    doc_top_topics = {}
    for doc in sorted_doc_topic.items():
        doc_top_topics[doc[0]] = [doc[1][index] for index in range(3)]

    topic_top_words = get_topics(corpora, num_topics, topic_word_dist, num_of_word_per_topic=50)

    # printing doc-topic -> topic-word connections
    print("Random documents with top topics:")
    for doc in doc_top_topics.items():
        id = doc[0]
        print(f"Document ID: {id}")
        print(doc2pre[id])
        print(color_words(list(itertools.chain.from_iterable([topic_top_words[x] for x, y in doc[1]])),
                          doc2raw[id]) + "\n")

        print("Top words in document top topics:")
        for topic_num, topic_percent in doc[1]:
            print(f"Topic ID/probability: {topic_num}/{'{:.2f}'.format(topic_percent)} {topic_top_words[topic_num]}")
        print()
