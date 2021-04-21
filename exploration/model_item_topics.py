import random

import preprocess.preprocessing as pre
from model.save import load_model
from gibbs_utility import get_topics


if __name__ == '__main__':
    corpora = pre.prepro_file_load("corpora", "full")
    doc2pre = pre.prepro_file_load('doc2pre_text', folder_name='full')
    doc2raw = pre.prepro_file_load('doc2raw_text', folder_name='full')
    model_path = "../model/models/90_0.01_0.1_author"
    model = load_model(model_path)
    model_type = model_path.split("_")[-1]
    num_topics = model.num_topics
    topic_word_dist = model.topic_word

    items = dict(enumerate(model.doc_topic))
    items = [(k, v) for k, v in items.items()]  # convert to list of (ID, topics)
    sample_items = random.sample(items, 5)  # sample 5 random documents
    # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
    sample_items = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items]

    # sort topics in sampled docs and keep top 3 topics
    sorted_item_topic = {}
    for item in sample_items:
        sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
    item_top_topics = {}
    for item in sorted_item_topic.items():
        item_top_topics[item[0]] = [item[1][index] for index in range(3)]

    topic_top_words = get_topics(corpora, num_topics, topic_word_dist)

    # printing item-topic -> topic-word connections
    if model_type == "standard":
        print("Random documents with top topics:")
    elif model_type == "category":
        print("Random categories with top topics:")
    elif model_type == "author":
        print("Random authors with top topics:")
    else:
        print("Model type not known!")
        exit()
    for item in item_top_topics.items():
        id = item[0]
        if model_type == "standard":
            print(f"Document ID: {id}")
            print(doc2pre[id])
            print(doc2raw[id] + "\n")
        else:
            print(f"{model_type.capitalize()} model ID: {id}")

        if model_type == "standard":
            print("Top words in document top topics:")
        else:
            print(f"Top words in {model_type} top topics:")
        for topic in item[1]:
            print(f"Topic ID/probability: {topic[0]}/{'{:.2f}'.format(topic[1])} {topic_top_words[topic[0]]}")
        print()
