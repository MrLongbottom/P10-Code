import math
import statistics

from model.save import load_model
from exploration_utility import row_distribution_normalization
import preprocess.preprocessing as pre


def calculate_author_similarity(author_topic, i: int, j: int):
    # sum the author similarities for all topics to get the symmetric KL divergence.
    # calculated based on equation 9 from the author topic paper
    sKL_divergence = 0
    for t in range(author_topic.shape[1]):
        sKL_divergence += author_topic[i, t] * math.log(author_topic[i, t] / author_topic[j, t]) + \
                          author_topic[j, t] * math.log(author_topic[j, t] / author_topic[i, t])
    return sKL_divergence


def calculate_author_similarities(author_topic):
    author_similarities = {}
    cur_diagonal = 0
    for i in range(author_topic.shape[0]):
        for j in range(author_topic.shape[0]):
            if j > cur_diagonal:  # only calculate half of the author matrix
                author_similarities[(i, j)] = calculate_author_similarity(author_topic, i, j)
        cur_diagonal += 1
    return author_similarities


if __name__ == '__main__':
    id2author = pre.prepro_file_load('id2author', folder_name='full')
    id2category = pre.prepro_file_load('id2category', folder_name='full')
    model_path = "../model/models/90_0.01_0.1_author"
    model = load_model(model_path)
    model_type = model_path.split("_")[-1]
    if model_type == "geographic":
        id2category = pre.prepro_file_load('id2category', folder_name='full_geographic')
    elif model_type == "topical":
        id2category = pre.prepro_file_load('id2category', folder_name='full_topical')

    # Get author-topic distribution and normalize
    author_topic = model.doc_topic
    author_topic = row_distribution_normalization(author_topic)

    author_similarities = calculate_author_similarities(author_topic)
    sorted_author_similarities = dict(sorted(author_similarities.items(), key=lambda item: item[1]))
    top_author_pairs = list(sorted_author_similarities.items())[:10]
    median_divergence = statistics.median(sorted_author_similarities.values())
    max_divergence = list(sorted_author_similarities.values())[-1]

    # Print gathered information
    print(f"Max KL divergence: {'{:.2f}'.format(max_divergence)}")
    print(f"Median KL divergence: {'{:.2f}'.format(median_divergence)}\n")
    if model_type == "author":
        print("Top 10 author pairs based on symmetric KL divergence:")
    else:
        print("Top 10 category pairs based on symmetric KL divergence:")
    for pair in top_author_pairs:
        authors = pair[0]
        if model_type == "author":
            print(f"KL divergence: {'{:.2f}'.format(pair[1])}, "
                  f"Authors: {id2author[authors[0]], id2author[authors[1]]}")
        else:
            print(f"KL divergence: {'{:.2f}'.format(pair[1])}, "
                  f"Categories: {id2category[authors[0]], id2category[authors[1]]}")
