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
    with open(filename, "r") as json_file:
        for index, json_obj in enumerate(json_file):
            try:
                data = json.loads(json_obj)
            except:
                print("loaded the json wrong")
                continue
            documents[data["id"]] = tokenize(data['headline'] + ' ' + data['body'])
            categories.append(data['category'])
            if index == 1000:
                break
    print('Loaded ' + str(len(documents)) + ' documents.')
    return dict(list(documents.items())[:1000]), categories[:1000]


if __name__ == '__main__':
    X = load_document_file("data/2017_data.json")
    print()
