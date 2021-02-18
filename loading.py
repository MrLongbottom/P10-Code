import json


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    categories = {}
    authors = {}
    taxonomies = {}
    with open(filename, "r") as json_file:
        for json_obj in json_file:
            try:
                data = json.loads(json_obj)
            except:
                print("problem with loading json")
                continue
            text = data['headline'] + ' ' + data['body']
            documents[data["id"]] = text
            categories[data["id"]] = data['category']
            authors[data["id"]] = data['author']
            taxonomies[data["id"]] = data['taxonomy']

    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents, categories, authors, taxonomies


if __name__ == '__main__':
    documents, categories, authors, taxonomies = load_document_file("data/2017_data.json")
    print()
