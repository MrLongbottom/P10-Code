import json
import re
import string

import nltk


def tokenize(text):
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return nltk.tokenize.word_tokenize(text_no_doublespace, 'danish')


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    categories = []
    with open(filename, "r+", encoding='utf-8') as json_file:
        for json_obj in json_file:
            try:
                data = json.loads(json_obj)
            except:
                print("loaded the json wrong")
                continue
            documents[data["id"]] = tokenize(data['headline'] + ' ' + data['body'])
            categories.append(data['category'])
    print('Loaded ' + str(len(documents)) + ' documents.')
    return documents, categories


if __name__ == '__main__':
    X = load_document_file("data/2017_data.json")
    print()
