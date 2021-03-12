import gensim
from tqdm import tqdm

import json
import utility
import itertools
import torch.sparse
import pickle


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
    id2doc_file = {num_id: name_id for num_id, name_id in enumerate(new_texts.keys())}

    # remove duplicates
    categories = {e: v for e, (k, v) in enumerate(categories.items()) if k not in bad_ids}
    authors = {e: v for e, (k, v) in enumerate(authors.items()) if k not in bad_ids}
    taxonomies = {e: v for e, (k, v) in enumerate(taxonomies.items()) if k not in bad_ids}

    # make value -> id mappings
    cat2id = {v: i for v, i in zip(list(set(categories.values())), range(len(set(categories.values()))))}
    auth2id = {v: i for v, i in zip(list(set(authors.values())), range(len(set(authors.values()))))}
    tax2id = {v: i for v, i in zip(list(set(taxonomies.values())), range(len(set(taxonomies.values()))))}

    # make doc_id -> meta_id mappings
    categories = {i: cat2id[v] for v, i in zip(categories.values(), range(len(categories)))}
    authors = {i: auth2id[v] for v, i in zip(authors.values(), range(len(authors)))}
    taxonomies = {i: tax2id[v] for v, i in zip(taxonomies.values(), range(len(taxonomies)))}

    texts = {e: v.replace('\n', '') for e, (k, v) in enumerate(texts.items()) if k not in bad_ids}
    if save:
        if printouts:
            print("Saving data mapping files")
        utility.save_dict_file('../' + paths['id2doc_name'], id2doc_file)
        utility.save_dict_file('../' + paths['id2raw_text'], texts)
        utility.save_dict_file('../' + paths['id2category'], {v: k for k, v in cat2id.items()})
        utility.save_dict_file('../' + paths['id2author'], {v: k for k, v in auth2id.items()})
        utility.save_dict_file('../' + paths['id2taxonomy'], {v: k for k, v in tax2id.items()})
        utility.save_dict_file('../' + paths['doc2category'], categories)
        utility.save_dict_file('../' + paths['doc2author'], authors)
        utility.save_dict_file('../' + paths['doc2taxonomy'], taxonomies)

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

    # clean up and construct & save files
    corpora.compactify()

    doc2id = []
    doc2 = []
    for doc in documents:
        doc_id = corpora.doc2idx(doc)
        doc2.append([doc[i] for i in range(len(doc)) if doc_id[i] != -1])
        doc2id.append([x for x in doc_id if x != -1])
    documents = doc2

    doc2bow = []
    for doc in documents:
        doc2bow.append(corpora.doc2bow(doc))

    doc_word_matrix = sparse_vector_document_representations(corpora, doc2bow)

    doc_cat_one_hot = torch.zeros((len(documents), len(cat2id)))
    for i, doc in enumerate(documents):
        doc_cat_one_hot[i, categories.get(i)] = 1

    if save:
        if printouts:
            print("Saving Corpora & Preprocessed Text")
        corpora.save('../' + paths['corpora'])
        with open('../' + paths['doc2bow'], "wb") as file:
            pickle.dump(doc2bow, file)
        with open('../' + paths['doc_word_matrix'], "wb") as file:
            pickle.dump(doc_word_matrix, file)
        with open('../' + paths['doc_cat_one_hot_matrix'], "wb") as file:
            pickle.dump(doc_cat_one_hot, file)
        utility.save_dict_file('../' + paths['id2word'], {v: k for k, v in corpora.token2id.items()})
        utility.save_dict_file('../' + paths['id2pre_text'], documents)
        utility.save_dict_file('../' + paths['doc2word_ids'], doc2id)

    if printouts:
        print('Preprocessing Finished.')
    return corpora, documents, doc2bow, doc_word_matrix


def sparse_vector_document_representations(corpora, doc2bow):
    doc_keys = list(itertools.chain.from_iterable
                    ([[[doc_id, word_id[0]] for word_id in doc2bow[doc_id]] for doc_id in range(len(doc2bow))]))
    doc_values = []
    for doc in tqdm(doc2bow):
        [doc_values.append(y) for x, y in doc]
    sparse_docs = torch.sparse.FloatTensor(torch.LongTensor(doc_keys).t(), torch.FloatTensor(doc_values),
                                          torch.Size([corpora.num_docs, len(corpora)]))
    return sparse_docs


def load_document_file(filename):
    print('Loading documents from "' + filename + '".')
    documents = {}
    categories = {}
    authors = {}
    taxonomies = {}

    with open(filename, "r", encoding='utf-8', errors='ignore') as json_file:
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


def prepro_file_load(file_name):
    paths = utility.load_dict_file("../paths.csv")
    if file_name not in paths:
        raise Exception('File name not in paths file')
    else:
        file_path = '../' + paths[file_name]
        if file_path[-4:] == '.csv':
            return utility.load_dict_file(file_path)
        elif file_path[-7:] == '.pickle':
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        else:
            return gensim.corpora.Dictionary.load(file_path)


if __name__ == '__main__':
    info = preprocessing(printouts=True, save=True)
    print('Finished Preprocessing')
