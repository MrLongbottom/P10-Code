import json


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    with open(filename, "r") as json_file:
        for json_obj in json_file:
            try:
                data = json.loads(json_obj)
            except:
                print("loaded the json wrong")
                continue
            meta = data['category']
            text = data['headline'] + ' ' + data['body']
            documents[data["id"]] = text

    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents, meta


if __name__ == '__main__':
    documents = load_document_file("data/2017_data.json")
    print()
