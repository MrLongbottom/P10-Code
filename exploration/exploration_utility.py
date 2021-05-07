import random

def find_id_from_value(dictionary: dict, value: str, printouts: bool = True):
    id = None
    for key, val in dictionary.items():
        if value == val:
            id = key

    if id is None:
        if printouts:
            print(f"Did not find the value '{value}' in the metadata")
        exit()
    else:
        if printouts:
            print(f"Found '{value}' with ID: {id}")
        return id


def get_category_ids_from_names(id2category: dict, names: list):
    category_ids = []
    for name in names:
        category_ids.append(find_id_from_value(id2category, name, printouts=False))
    return category_ids


def row_distribution_normalization(matrix):
    normalized_matrix = matrix
    for row in range(normalized_matrix.shape[0]):
        row_sum = sum(normalized_matrix[row])
        normalized_matrix[row] = [val / row_sum for val in normalized_matrix[row]]
    return normalized_matrix


def sample_and_sort_items(model, num_items: int = 5, num_top_topics: int = 3, item_id=None):
    item_top_topics = {}
    if model.name == "MultiModel":
        items = dict(enumerate(model.feature_topic))
        items = [(k, v) for k, v in items.items()]  # convert to list of (ID, topics)
        items2 = dict(enumerate(model.feature2_topic))
        items2 = [(k, v) for k, v in items2.items()]  # convert to list of (ID, topics)

        # Random sampling of the two features
        sample_items = random.sample(items, num_items)
        sample_items2 = random.sample(items2, num_items)
        # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
        sample_items = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items]
        sample_items2 = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items2]

        # sort topics in sampled docs and keep top 'num_top_topics' topics
        sorted_item_topic = {}
        sorted_item_topic_feature = {}
        sorted_item_topic_feature2 = {}
        for item in sample_items:
            sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
        for item in sorted_item_topic.items():
            sorted_item_topic_feature[item[0]] = [item[1][index] for index in range(num_top_topics)]
        item_top_topics[0] = sorted_item_topic_feature
        sorted_item_topic.clear()
        for item in sample_items2:
            sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
        for item in sorted_item_topic.items():
            sorted_item_topic_feature2[item[0]] = [item[1][index] for index in range(num_top_topics)]
        item_top_topics[1] = sorted_item_topic_feature2
    else:
        items = dict(enumerate(model.doc_topic))
        items = [(k, v) for k, v in items.items()]  # convert to list of (ID, topics)

        if item_id is None:
            sample_items = random.sample(items, num_items)  # sample 'num_items' random documents/categories/authors/etc
        else:  # sample from a single ID
            sample_items = [(k, v) for k, v in items if k == item_id]
        # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
        sample_items = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items]

        # sort topics in sampled docs and keep top 'num_top_topics' topics
        sorted_item_topic = {}
        for item in sample_items:
            sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
        for item in sorted_item_topic.items():
            item_top_topics[item[0]] = [item[1][index] for index in range(num_top_topics)]
    return item_top_topics


def get_metadata_document_ids(doc2meta, meta_id):
    document_ids = []
    for key, val in doc2meta.items():
        if type(val) is list:  # if it's a list, then it's a taxonomy
            if meta_id in val:
                document_ids.append(key)
        else:
            if meta_id == val:
                document_ids.append(key)
    return document_ids
