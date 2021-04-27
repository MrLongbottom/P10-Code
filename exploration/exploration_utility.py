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
    items = dict(enumerate(model.doc_topic))
    items = [(k, v) for k, v in items.items()]  # convert to list of (ID, topics)
    if item_id is None:
        sample_items = random.sample(items, num_items)  # sample 'num_items' random documents/categories/authors/etc.
    else:  # sample from a single ID
        sample_items = [(k, v) for k, v in items if k == item_id]
    # convert topic probabilities to (ID, probability) pairs to know IDs after sorting
    sample_items = [(item[0], [(index, topic) for index, topic in enumerate(item[1])]) for item in sample_items]

    # sort topics in sampled docs and keep top 'num_top_topics' topics
    sorted_item_topic = {}
    for item in sample_items:
        sorted_item_topic[item[0]] = sorted(item[1], key=lambda topic: topic[1], reverse=True)
    item_top_topics = {}
    for item in sorted_item_topic.items():
        item_top_topics[item[0]] = [item[1][index] for index in range(num_top_topics)]

    return item_top_topics