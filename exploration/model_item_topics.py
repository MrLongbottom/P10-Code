import preprocess.preprocessing as pre
from model.save import load_model
from exploration_utility import sample_and_sort_items
from gibbs_utility import get_topics


def print_latex_table(item_top_topics, topic_top_words, id2meta):
    number_of_items = len(item_top_topics)

    latex_table = []
    # generate table based on item names and their topics/words
    for index, item_id in enumerate(item_top_topics.keys()):
        if index == 0:
            latex_table.append(id2meta[item_id])
        else:
            latex_table[0] += " & " + id2meta[item_id]

    latex_table.append("\\midrule")

    for index, item_topics in enumerate(item_top_topics.values()):
        if index == 0:
            latex_table.append(f"Topic {item_topics[0][0]}")
        else:
            latex_table[2] += " & Topic " + str(item_topics[0][0])

    latex_table.append("\\midrule")

    for index, item_topics in enumerate(item_top_topics.values()):
        if index == 0:
            latex_table.append(top_words_to_table_cell(topic_top_words[item_topics[0][0]]))
        else:
            latex_table[4] += " & " + top_words_to_table_cell(topic_top_words[item_topics[0][0]])

    # add required end of row characters and print row
    for index, row in enumerate(latex_table):
        if not (index == 1 or index == 3):
            latex_table[index] += " \\\\"
        print(latex_table[index])


def top_words_to_table_cell(top_words):
    table_cell = "\\makecell{"
    for index, word in enumerate(top_words):
        if index == len(top_words) - 1:
            table_cell += word
        else:
            table_cell += f"{word} \\\\ "
    table_cell += "}"
    return table_cell


if __name__ == '__main__':
    corpora = pre.prepro_file_load("corpora", "full")
    doc2pre = pre.prepro_file_load('doc2pre_text', folder_name='full')
    doc2raw = pre.prepro_file_load('doc2raw_text', folder_name='full')
    id2category = pre.prepro_file_load('id2category', folder_name='full')
    id2author = pre.prepro_file_load('id2author', folder_name='full')
    model_path = "../model/models/90_0.01_0.1_author"
    model_type = model_path.split("_")[-1]
    model = load_model(model_path) if model_type != "MultiModel" else load_model(model_path, multi=True)
    num_topics = model.num_topics
    topic_word_dist = model.topic_word
    if model_type == "geographic":
        id2category = pre.prepro_file_load('id2category', folder_name='full_geographic')
    elif model_type == "topical":
        id2category = pre.prepro_file_load('id2category', folder_name='full_topical')
    model_type = "category" if model_type == "geographic" or model_type == "topical" else model_type

    item_top_topics = sample_and_sort_items(model, num_items=7)
    topic_top_words = get_topics(corpora, num_topics, topic_word_dist)

    # printing item-topic -> topic-word connections
    if model_type == "standard":
        print("Random documents with top topics:")
    elif model_type == "category":
        print("Random categories with top topics:")
    elif model_type == "author":
        print("Random authors with top topics:")
    elif model_type == "MultiModel":
        print("Random categories and authors with top topics:")
    else:
        print(f"Model type '{model_type}' not known!")
        exit()

    for item in item_top_topics.items():
        if model_type == "MultiModel":
            for feature in item[1].items():
                id = feature[0]
                print(f"{'Author' if item[0] == 0 else 'Category'} "
                      f"ID: {id} ({id2author[id] if item[0] == 0 else id2category[id]})")
                print(f"Top words in {'author' if item[0] == 0 else 'category'} top topics:")
                for topic in feature[1]:
                    print(f"Topic ID/probability: {topic[0]}/{'{:.2f}'.format(topic[1])} {topic_top_words[topic[0]]}")
                print()
        else:
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

    print()
    id2meta = id2author if model_type == "author" else id2category
    print_latex_table(item_top_topics, topic_top_words, id2meta)
