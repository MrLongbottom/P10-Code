import random

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

    id2meta = pre.prepro_file_load('id2category', folder_name='full')
    doc2meta = pre.prepro_file_load('doc2category', folder_name='full')
    value = "26. Frederik"
    # id2meta = pre.prepro_file_load('id2author', folder_name='full')
    # doc2meta = pre.prepro_file_load('doc2author', folder_name='full')
    # value = "Villy Dall"

    metaID = find_id_from_value(id2meta, value)

    documentIDs = []
    for key, val in doc2meta.items():
        if metaID == val:
            documentIDs.append(key)

    documents = {}
    documentsRaw = {}
    docFileNames = {}
    for docID in documentIDs:
        documents[docID] = doc2pre.get(docID)
        documentsRaw[docID] = doc2raw.get(docID)
        docFileNames[docID] = id2doc.get(docID)

    print(f"{len(documents)} documents found\n")
    print("10 random documents:")
    for count in range(10):
        id = random.choice(documentIDs)
        print(id)
        print(documents[id])
        print(documentsRaw[id] + "\n")
