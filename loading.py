import json


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    categories = {}
    with open(filename, "r") as json_file:
        for json_obj in json_file:
            try:
                data = json.loads(json_obj)
            except:
                print("problem with loading json")
                continue
            category = data['category']
            text = data['headline'] + ' ' + data['body']
            documents[data["id"]] = text
            categories[data["id"]] = category

    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents, meta


if __name__ == '__main__':
    documents, meta = load_document_file("data/2017_data.json")
    print()
