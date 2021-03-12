import gensim
from tqdm import tqdm
import json
import utility
import itertools
import torch.sparse
import pickle
import numpy as np


def preprocessing(json_file, printouts=False, save=True, folder_name=""):
    paths = utility.load_dict_file("../paths.csv")

    # load data file
    if printouts:
        print("Loading dataset")
    texts, categories, authors, taxonomies = load_document_file('../' + paths[json_file])

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
        utility.save_dict_file('../' + update_path(paths['id2doc'], folder_name), id2doc_file)
        utility.save_dict_file('../' + update_path(paths['doc2raw_text'], folder_name), texts)
        utility.save_dict_file('../' + update_path(paths['id2category'], folder_name),
                               {v: k for k, v in cat2id.items()})
        utility.save_dict_file('../' + update_path(paths['id2author'], folder_name), {v: k for k, v in auth2id.items()})
        utility.save_dict_file('../' + update_path(paths['id2taxonomy'], folder_name),
                               {v: k for k, v in tax2id.items()})
        utility.save_dict_file('../' + update_path(paths['doc2category'], folder_name), categories)
        utility.save_dict_file('../' + update_path(paths['doc2author'], folder_name), authors)
        utility.save_dict_file('../' + update_path(paths['doc2taxonomy'], folder_name), taxonomies)

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

    doc2bow = make_doc2bow(corpora, documents)

    # Construct the doc word matrix by using mem map
    shape = (corpora.num_docs, len(corpora))
    mm_doc_word_matrix = save_memmap_matrix("doc_word_matrix", doc2bow, shape)

    # Sparse tensor format
    doc_word_matrix = sparse_vector_document_representations(corpora, doc2bow)

    if save:
        if printouts:
            print("Saving Corpora & Preprocessed Text")
        corpora.save('../' + paths['corpora'])
        with open('../' + update_path(paths['doc2bow'], folder_name), "wb") as file:
            pickle.dump(doc2bow, file)
        with open('../' + paths['doc_word_matrix_sparse'], "wb") as file:
            pickle.dump(doc_word_matrix, file)
        utility.save_dict_file('../' + paths['id2word'], {v: k for k, v in corpora.token2id.items()})
        utility.save_dict_file('../' + paths['doc2pre_text'], documents)
        utility.save_dict_file('../' + paths['doc2word'], doc2id)

    if printouts:
        print('Preprocessing Finished.')
    return corpora, documents, doc2bow, mm_doc_word_matrix


def make_doc2bow(corpora, documents):
    doc2bow = []
    for doc in documents:
        doc2bow.append(corpora.doc2bow(doc))
    return doc2bow


def update_path(path, folder_name):
    if folder_name == "":
        return path
    path = path.split('/')
    path.insert(len(path) - 1, folder_name)
    path = "/".join(path)
    return path


def sparse_vector_document_representations(corpora, doc2bow):
    doc_keys = list(itertools.chain.from_iterable
                    ([[[doc_id, word_id[0]] for word_id in doc2bow[doc_id]] for doc_id in range(len(doc2bow))]))
    doc_values = []
    for doc in tqdm(doc2bow):
        [doc_values.append(y) for x, y in doc]
    sparse_docs = torch.sparse.FloatTensor(torch.LongTensor(doc_keys).t(), torch.FloatTensor(doc_values),
                                           torch.Size([corpora.num_docs, len(corpora)]))
    return sparse_docs


def save_memmap_matrix(name: str, doc2bow, shape: (int, int)):
    paths = utility.load_dict_file("../paths.csv")
    mm_doc_word_matrix = np.memmap('../' + paths[name], dtype=np.int, mode='w+', shape=shape)
    for doc_index, words in enumerate(doc2bow):
        for word_index, word_count in words:
            mm_doc_word_matrix[doc_index, word_index] = word_count
    return mm_doc_word_matrix


def load_memmap_matrix(name: str):
    corpora = prepro_file_load("corpora")
    shape = (corpora.num_docs, len(corpora))
    return np.memmap(f'{name}', dtype=np.int, shape=shape)


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


def prepro_file_load(file_name, folder_name=None):
    paths = utility.load_dict_file("../paths.csv")
    if file_name not in paths:
        raise Exception('File name not in paths file')
    else:
        file_path = '../' + paths[file_name]
        if folder_name is not None:
            file_path = update_path(file_path, folder_name)
        if file_path[-4:] == '.csv':
            return utility.load_dict_file(file_path)
        elif file_path[-7:] == '.pickle':
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        else:
            return gensim.corpora.Dictionary.load(file_path)


if __name__ == '__main__':
    info = preprocessing(json_file='2017_json', printouts=True, save=True)
    print('Finished Preprocessing')
