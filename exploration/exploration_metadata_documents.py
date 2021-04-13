import random
from collections import Counter

import preprocess.preprocessing as pre


def find_id_from_value(dictionary: dict, value: str):
    id = None
    for key, val in dictionary.items():
        if value == val:
            id = key

    if id is None:
        print(f"Did not find the value '{value}' in the metadata")
        exit()
    else:
        print(f"Found '{value}' with ID: {id}")
        return id


def print_meta_document_set_info(doc2found_meta: dict, id2meta: dict, meta_name: str):
    unique_meta = set(doc2found_meta.values())
    unique_meta_count = Counter(doc2found_meta.values())
    meta_count_pairs = [(author, unique_meta_count[author]) for author in unique_meta]
    sorted_meta_count_pairs = sorted(meta_count_pairs, key=lambda pair: pair[1], reverse=True)
    print(f"{len(unique_meta)} unique {meta_name}: ")
    print(f"{[(id2meta[pair[0]], pair[1]) for pair in sorted_meta_count_pairs]}\n")


if __name__ == '__main__':
    doc2pre = pre.prepro_file_load('doc2pre_text', folder_name='full')
    doc2raw = pre.prepro_file_load('doc2raw_text', folder_name='full')
    id2doc = pre.prepro_file_load('id2doc', folder_name='full')
    doc2author = pre.prepro_file_load('doc2author', folder_name='full')
    doc2category = pre.prepro_file_load('doc2category', folder_name='full')

    id2category = pre.prepro_file_load('id2category', folder_name='full')
    # doc2meta = pre.prepro_file_load('doc2category', folder_name='full')
    # value = "26. Frederik"
    id2author = pre.prepro_file_load('id2author', folder_name='full')
    doc2meta = pre.prepro_file_load('doc2author', folder_name='full')
    value = "System Administrator"

    metaID = find_id_from_value(id2author, value)

    # get document IDs for documents with the given metadata
    documentIDs = []
    for key, val in doc2meta.items():
        if metaID == val:
            documentIDs.append(key)

    documents = {}
    documentsRaw = {}
    docAuthors = {}
    docCategories = {}
    docFileNames = {}
    # get data based on document IDs
    for docID in documentIDs:
        documents[docID] = doc2pre[docID]
        documentsRaw[docID] = doc2raw[docID]
        docAuthors[docID] = doc2author[docID]
        docCategories[docID] = doc2category[docID]
        docFileNames[docID] = id2doc[docID]

    # document list information
    print(f"{len(documents)} documents found\n")
    print_meta_document_set_info(docAuthors, id2author, "authors")
    print_meta_document_set_info(docCategories, id2category, "categories")

    # random examples of documents with metadata information
    print("Random documents:")
    sampleIDs = random.sample(documentIDs, len(documentIDs))
    for count in range(10):
        if count == len(sampleIDs):
            break
        id = sampleIDs[count]
        print(f"ID: {id}")
        print(f"Author: {id2author[docAuthors[id]]}")
        print(f"Category: {id2category[docCategories[id]]}")
        print(f"File name: {docFileNames[id]}")
        print(documents[id])
        print(documentsRaw[id] + "\n")
