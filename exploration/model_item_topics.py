import itertools
from typing import List

from tqdm import tqdm

import preprocess.preprocessing as pre
from gibbs_utility import get_topics
from model.save import load_model


def color_words(words: List[str], text: str, color: str):
    for word in words:
        text = text.replace(" " + word + " ", " \\" + "colorbox" + "{" + color + "}" + "{" + word + "} ")
    return text


if __name__ == '__main__':
    corpora = pre.prepro_file_load("corpora")
    doc2pre = pre.prepro_file_load('doc2pre_text')
    doc2raw = pre.prepro_file_load('doc2raw_text')
    id2category = pre.prepro_file_load('id2category')
    id2author = pre.prepro_file_load('id2author')

    # specific document
    doc_id = 38668
    doc_author = pre.prepro_file_load("doc2author")[doc_id]
    doc_category = pre.prepro_file_load("doc2category")[doc_id]

    models = ['standard', 'author', 'category']
    model_words = {}
    for model_name in tqdm(models):
        model_path = f"../model/90_0.01_0.1_{model_name}"
        model = load_model(model_path)
        model_type = model_path.split("_")[-1]
        num_topics = model.num_topics
        topic_word_dist = model.topic_word

        items = dict(enumerate(model.doc_topic))
        items = [(k, v) for k, v in items.items()]  # convert to list of (ID, topics)
        sample_items = []
        if model_type == "standard":
            sample_items = [items[doc_id]]
        elif model_type == "category":
            sample_items = [items[doc_category]]
        elif model_type == "author":
            sample_items = [items[doc_author]]
        else:
            print("Model type not known!")
            exit()
        # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
        sample_items = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items]

        # sort topics in sampled docs and keep top 3 topics
        sorted_item_topic = {}
        for item in sample_items:
            sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
        item_top_topics = {}
        for item in sorted_item_topic.items():
            item_top_topics[item[0]] = [item[1][index] for index in range(3)]

        topic_top_words = get_topics(corpora, num_topics, topic_word_dist, num_of_word_per_topic=200)

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
            elif model_type == "category":
                print(f"{model_type.capitalize()} ID: {id} ({id2category[id]})")

            elif model_type == "author":
                print(f"{model_type.capitalize()} ID: {id} ({id2author[id]})")

            if model_type == "standard":
                print("Top words in document top topics:")
            else:
                print(f"Top words in {model_type} top topics:")
            for topic in item[1]:
                print(f"Topic ID/probability: {topic[0]}/{'{:.2f}'.format(topic[1])} {topic_top_words[topic[0]]}")
            print()
            model_words[model_name] = list(itertools.chain.from_iterable([topic_top_words[x] for x, y in item[1]]))

    text = doc2raw[doc_id]
    standard_words, cat_words, author_words = [[x for x in model_words[name] if x in text] for name in models]
    all_encountered_words = list(set(list(
        itertools.chain.from_iterable([[x for x in model_words[name] if x in text] for name in models]))))
    for word in all_encountered_words:
        if word in standard_words and word in cat_words and word in author_words:
            text = color_words([word], text, "Peach")
        elif word in standard_words and word in cat_words:
            text = color_words([word], text, "Peach")
        elif word in standard_words and word in author_words:
            text = color_words([word], text, "Peach")
        elif word in standard_words:
            text = color_words([word], text, "Goldenrod")
        elif word in cat_words:
            text = color_words([word], text, "LimeGreen")
        elif word in author_words:
            text = color_words([word], text, "Aquamarine")
    print(text)
