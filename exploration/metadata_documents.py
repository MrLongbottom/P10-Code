import random
import itertools
from collections import Counter

from exploration_utility import find_id_from_value, sample_and_sort_items
from model.save import load_model
from gibbs_utility import get_topics
import preprocess.preprocessing as pre


def print_meta_document_set_info(doc2found_meta: dict, id2meta: dict, meta_name: str):
    if meta_name == "taxonomies":
        doc2found_meta_list = list(itertools.chain.from_iterable(list(doc2found_meta.values())))
        unique_meta = set(doc2found_meta_list)
        unique_meta_count = Counter(doc2found_meta_list)
    else:
        unique_meta = set(doc2found_meta.values())
        unique_meta_count = Counter(doc2found_meta.values())
    meta_count_pairs = [(id, unique_meta_count[id]) for id in unique_meta]
    sorted_meta_count_pairs = sorted(meta_count_pairs, key=lambda pair: pair[1], reverse=True)
    print(f"{len(unique_meta)} unique {meta_name}: ")
    print(f"{[(id2meta[pair[0]], pair[1]) for pair in sorted_meta_count_pairs]}\n")


def print_metadata_documents(metadata_type: str, metadata_name: str, sample_size: int = 10, print_top_topics=False):
    # load necessary data
    doc2pre = pre.prepro_file_load('doc2pre_text', folder_name='full')
    doc2raw = pre.prepro_file_load('doc2raw_text', folder_name='full')
    id2doc = pre.prepro_file_load('id2doc', folder_name='full')
    doc2author = pre.prepro_file_load('doc2author', folder_name='full')
    doc2category = pre.prepro_file_load('doc2category', folder_name='full')
    doc2taxonomy = pre.prepro_file_load('doc2taxonomy', folder_name='full')
    id2category = pre.prepro_file_load('id2category', folder_name='full')
    id2author = pre.prepro_file_load('id2author', folder_name='full')
    id2taxonomy = pre.prepro_file_load('id2taxonomy', folder_name='full')

    if metadata_type == "category":
        doc2meta = pre.prepro_file_load('doc2category', folder_name='full')
        id2meta = id2category
    elif metadata_type == "author":
        doc2meta = pre.prepro_file_load('doc2author', folder_name='full')
        id2meta = id2author
    elif metadata_type == "taxonomy":
        doc2meta = pre.prepro_file_load('doc2taxonomy', folder_name='full')
        id2meta = id2taxonomy
    else:
        print(f"'{metadata_type}' not found!")
        exit()

    # get metadata ID from name (examples: '26. Frederik', 'System Administrator', 'EMNER')
    metaID = find_id_from_value(id2meta, metadata_name)

    # get document IDs for documents with the given metadata
    documentIDs = []
    for key, val in doc2meta.items():
        if type(val) is list:  # if it's a list, then it's a taxonomy
            if metaID in val:
                documentIDs.append(key)
        else:
            if metaID == val:
                documentIDs.append(key)

    documents = {}
    documentsRaw = {}
    docAuthors = {}
    docCategories = {}
    docTaxonomies = {}
    docFileNames = {}
    # get data based on document IDs
    for docID in documentIDs:
        documents[docID] = doc2pre[docID]
        documentsRaw[docID] = doc2raw[docID]
        docAuthors[docID] = doc2author[docID]
        docCategories[docID] = doc2category[docID]
        docTaxonomies[docID] = doc2taxonomy[docID]
        docFileNames[docID] = id2doc[docID]

    # document set information
    print(f"{len(documents)} documents found\n")
    print_meta_document_set_info(docAuthors, id2author, "authors")
    print_meta_document_set_info(docCategories, id2category, "categories")
    print_meta_document_set_info(docTaxonomies, id2taxonomy, "taxonomies")

    # random examples of documents with metadata information
    if print_top_topics:
        corpora = pre.prepro_file_load("corpora", "full")
        model_path = "../model/models/90_0.01_0.1_category"
        model = load_model(model_path)
        num_topics = model.num_topics
        topic_word_dist = model.topic_word
        topic_top_words = get_topics(corpora, num_topics, topic_word_dist)

    print("Random documents:")
    sampleIDs = random.sample(documentIDs, len(documentIDs))
    for count in range(sample_size):
        if count == len(sampleIDs):
            break
        id = sampleIDs[count]
        print(f"ID: {id}")
        print(f"Author: {id2author[docAuthors[id]]}")
        print(f"Category: {id2category[docCategories[id]]}")
        print(f"Taxonomy: {[id2taxonomy[taxID] for taxID in docTaxonomies[id]]}")
        print(f"File name: {docFileNames[id]}")
        print(documents[id])
        print(documentsRaw[id] + "\n")
        if print_top_topics:
            item_top_topics = sample_and_sort_items(model, item_id=docCategories[id])
            print(f"Top words in category top topics:")
            for item in item_top_topics.items():
                for topic in item[1]:
                    print(f"Topic ID/probability: {topic[0]}/{'{:.2f}'.format(topic[1])} {topic_top_words[topic[0]]}")
            print()


if __name__ == '__main__':
    print_metadata_documents("category", "Sport-avis")
    # print_metadata_documents("author", "System Administrator")
    # print_metadata_documents("taxonomy", "EMNER", print_top_topics=True)
