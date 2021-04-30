import numpy as np

from model.save import load_model
from exploration_utility import row_distribution_normalization, get_category_ids_from_names
from gibbs_utility import get_topics
import preprocess.preprocessing as pre


def delete_rows_and_sort(matrix, ids):
    filtered_matrix = np.delete(matrix, ids, axis=0)
    summed_matrix = np.sum(filtered_matrix, axis=0)
    summed_dict = {i: v for i, v in enumerate(summed_matrix)}
    sorted_summed_dict = dict(sorted(summed_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_summed_dict


def print_top_topics_geographic_and_topical(num_top_topics: int = 20):
    model_path = "../model/models/90_0.01_0.1_category"
    model = load_model(model_path)
    corpora = pre.prepro_file_load("corpora", "full")
    id2category = pre.prepro_file_load('id2category', folder_name='full')
    num_topics = model.num_topics
    topic_word_dist = model.topic_word

    # names of categories that are based on a geographic location
    geographic_category_names = ["Frederikshavn-avis", "Morsø Debat", "Morsø-avis", "Rebild-avis", "Brønderslev-avis",
                                 "Thisted-avis", "Jammerbugt-avis", "Vesthimmerland-avis", "Hjørring-avis",
                                 "Aalborg-avis", "Morsø Sport", "Thisted sport", "Mariagerfjord-avis", "Udland-avis"]
    geographic_category_ids = get_category_ids_from_names(id2category, geographic_category_names)
    # categories not based on geographic locations are closer to real topics
    topical_category_ids = [id for id in id2category.keys() if id not in geographic_category_ids]
    category_topic = model.doc_topic
    category_topic = row_distribution_normalization(category_topic)

    # separate the geographic and topical topic distributions and sort on the topics' summed distribution values
    sorted_geographic = delete_rows_and_sort(category_topic, topical_category_ids)
    sorted_topical = delete_rows_and_sort(category_topic, geographic_category_ids)

    # look for topic ID appearances in both top topic lists and unique appearances
    top_multiple_topics = []
    for index in range(num_top_topics):
        cur_topic = list(sorted_geographic.keys())[index]
        if cur_topic in list(sorted_topical.keys())[:num_top_topics]:
            top_multiple_topics.append(cur_topic)
    top_unique_topics = list(
        set(list(sorted_geographic.keys())[:num_top_topics] + list(sorted_topical.keys())[:num_top_topics]))
    for topic in top_multiple_topics:
        top_unique_topics.remove(topic)
    for index, topic in enumerate(top_unique_topics):
        if topic in list(sorted_geographic.keys())[:num_top_topics]:
            top_unique_topics[index] = ("geographic", top_unique_topics[index])
        else:
            top_unique_topics[index] = ("topical", top_unique_topics[index])

    topic_top_words = get_topics(corpora, num_topics, topic_word_dist)

    # print observations
    print("Top topics for only geographic categories:")
    for index in range(num_top_topics):
        topic = list(sorted_geographic.items())[index]
        print(f"{'{:.2f}'.format(topic[1])}, {topic[0]}: {topic_top_words[topic[0]]}")

    print()
    print("Top topics for only topical categories:")
    for index in range(num_top_topics):
        topic = list(sorted_topical.items())[index]
        print(f"{'{:.2f}'.format(topic[1])}, {topic[0]}: {topic_top_words[topic[0]]}")

    print()
    print("Appears in both lists:")
    print(f"{len(top_multiple_topics)}/{num_top_topics}")
    for topic in top_multiple_topics:
        print(f"{topic}: {topic_top_words[topic]}")

    print()
    print("Unique topics:")
    for topic in top_unique_topics:
        print(f"{topic}: {topic_top_words[topic[1]]}")


if __name__ == '__main__':
    print_top_topics_geographic_and_topical()
