import gensim
from tqdm import tqdm
import json
import utility


def preprocessing(printouts=False, save=True):
    paths = utility.load_dict_file("../paths.csv")

    # load data file
    if printouts:
        print("Loading dataset")
    texts, categories, authors, taxonomies = load_document_file('../' + paths['2017_json'])

    # removing duplicates from dictionaries
    rev = {v: k for k, v in texts.items()}
    new_texts = {v: k for k, v in rev.items()}
    bad_ids = [x for x in texts.keys() if x not in new_texts.keys()]
    categories = {k: v for k, v in categories.items() if k not in bad_ids}
    authors = {k: v for k, v in authors.items() if k not in bad_ids}
    taxonomies = {k: v for k, v in taxonomies.items() if k not in bad_ids}
    texts = new_texts
    if save:
        if printouts:
            print("Saving data mapping files")
        utility.save_dict_file('../' + paths['id2raw_text'], texts)
        utility.save_dict_file('../' + paths['id2category'], categories)
        utility.save_dict_file('../' + paths['id2author'], authors)
        utility.save_dict_file('../' + paths['id2taxonomy'], taxonomies)

    # tokenize (document token generators)
    if printouts:
        print("Tokenization")
    documents = [list(gensim.utils.tokenize(x, lowercase=True, deacc=True, encoding='utf-8')) for x in
                 tqdm(texts.values())]

    # Build Corpora object
    if printouts:
        print("Building Corpora")
    corpora = gensim.corpora.Dictionary(documents)

    # Extreme filter
    if printouts:
        print("Filtering out extreme words")
    corpora.filter_extremes(no_below=10, no_above=0.1)

    # clean and save corpora
    corpora.compactify()
    if save:
        if printouts:
            print("Saving Corpora & Preprocessed Text")
        corpora.save('../' + paths['corpora'])
        # TODO save down id2word_id mapping and id2words mapping

    if printouts:
        print('Preprocessing Finished.')
    return corpora, documents


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
    preprocessing(printouts=True, save=True)
