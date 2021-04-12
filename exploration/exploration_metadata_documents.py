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
    value = "Villy Dall"

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

    uniqueAuthors = set(docAuthors.values())
    uniqueAuthorCount = Counter(docAuthors.values())
    authorCountPairs = [(author, uniqueAuthorCount[author]) for author in uniqueAuthors]
    sortedAuthorCountPairs = sorted(authorCountPairs, key=lambda pair: pair[1], reverse=True)
    print(f"{len(uniqueAuthors)} unique authors: ")
    print(f"{[(id2author[pair[0]], pair[1]) for pair in sortedAuthorCountPairs]}\n")

    uniqueCategories = set(docCategories.values())
    uniqueCategoryCount = Counter(docCategories.values())
    categoryCountPairs = [(category, uniqueCategoryCount[category]) for category in uniqueCategories]
    sortedCategoryCountPairs = sorted(categoryCountPairs, key=lambda pair: pair[1], reverse=True)
    print(f"{len(uniqueCategories)} unique categories: ")
    print(f"{[(id2category[pair[0]], pair[1]) for pair in sortedCategoryCountPairs]}\n")

    # random examples of documents with ID, author, preprocessed data, and raw data
    print("10 random documents:")
    for count in range(10):
        id = random.choice(documentIDs)
        print(f"ID: {id}")
        print(f"Author: {id2author[docAuthors[id]]}")
        print(f"Category: {id2category[docCategories[id]]}")
        print(f"File name: {docFileNames[id]}")
        print(documents[id])
        print(documentsRaw[id] + "\n")
