import random

import preprocess.preprocessing as pre
from model.save import load_model
from gibbs_utility import get_topics


if __name__ == '__main__':
    corpora = pre.prepro_file_load("corpora", "full")
    doc2pre = pre.prepro_file_load('doc2pre_text', folder_name='full')
    doc2raw = pre.prepro_file_load('doc2raw_text', folder_name='full')
    model = load_model("../model/models/90_0.01_0.1_standard")
    num_topics = model["num_topic"]
    topic_word_dist = model["topic_word"]

    docs = dict(enumerate(model["doc_topic"]))
    docs = [(k, v) for k, v in docs.items()]  # convert to list of (ID, topics)
    sample_docs = random.sample(docs, 5)  # sample 5 random documents
    # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
    sample_docs = [(doc[0], [(index, topic) for index, topic in enumerate(doc[1])]) for doc in sample_docs]

    # sort topics in sampled docs and keep top 3 topics
    sorted_doc_topic = {}
    for doc in sample_docs:
        sorted_doc_topic[doc[0]] = sorted(doc[1], key=lambda topic: topic[1], reverse=True)
    doc_top_topics = {}
    for doc in sorted_doc_topic.items():
        doc_top_topics[doc[0]] = [doc[1][index] for index in range(3)]

    topic_top_words = get_topics(corpora, num_topics, topic_word_dist)

    # printing doc-topic -> topic-word connections
    print("Random documents with top topics:")
    for doc in doc_top_topics.items():
        id = doc[0]
        print(f"Document ID: {id}")
        print(doc2pre[id])
        print(doc2raw[id] + "\n")

        print("Top words in document top topics:")
        for topic in doc[1]:
            print(f"Topic ID/probability: {topic[0]}/{'{:.2f}'.format(topic[1])} {topic_top_words[topic[0]]}")
        print()
