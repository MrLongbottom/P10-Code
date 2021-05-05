import gensim
from tqdm import tqdm
import Stemmer  # PyStemmer

import json
import utility
import itertools
import torch.sparse
import pickle


def preprocessing(json_file, printouts=False, save=True, folder_name="", stem=False, cat_names=None):
    paths = utility.load_dict_file("../paths.csv")

    # load data file
    if printouts:
        print("Loading dataset")
    texts, categories, authors, taxonomies = load_document_file('../' + paths[json_file])

    if cat_names is not None:
        if printouts:
            print("Filtering dataset")
        filtered_docs = [k for (k, v) in categories.items() if v in cat_names]
        texts = {k: v for (k, v) in texts.items() if k not in filtered_docs}
        categories = {k: v for (k, v) in categories.items() if k not in filtered_docs}
        authors = {k: v for (k, v) in authors.items() if k not in filtered_docs}
        taxonomies = {k: v for (k, v) in taxonomies.items() if k not in filtered_docs}

    # removing duplicates from dictionaries
    if printouts:
        print("Removing duplicates")
    rev = {v: k for k, v in texts.items()}
    new_texts = {v: k for k, v in rev.items()}
    bad_ids = [x for x in texts.keys() if x not in new_texts.keys()]
    id2doc_file = {num_id: name_id for num_id, name_id in enumerate(new_texts.keys())}
    texts = {e: v.replace('\n', '') for e, (k, v) in enumerate(new_texts.items()) if k not in bad_ids}
    categories = {e: v for e, (k, v) in enumerate(categories.items()) if k not in bad_ids}
    authors = {e: v for e, (k, v) in enumerate(authors.items()) if k not in bad_ids}
    taxonomies = {e: v for e, (k, v) in enumerate(taxonomies.items()) if k not in bad_ids}

    if stem:
        if printouts:
            print("Stemming")
        stemmer = Stemmer.Stemmer('danish')
        stemmed_texts = []
        for doc in tqdm(texts.values()):
            stemmed_text = stemmer.stemWords(doc.split())
            stemmed_texts.append(" ".join(stemmed_text))
        texts = {i: v for i, v in enumerate(stemmed_texts)}

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
    corpora.filter_extremes(no_below=10, no_above=0.10)

    # clean up and construct & save files
    corpora.compactify()

    doc2id = []
    doc2 = []
    empty_docs = []
    for id, doc in enumerate(documents):
        doc_id = corpora.doc2idx(doc)
        if all([x == -1 for x in doc_id]):
            empty_docs.append(id)
            continue
        doc2.append([doc[i] for i in range(len(doc)) if doc_id[i] != -1])
        doc2id.append([x for x in doc_id if x != -1])
    documents = doc2

    # remove info based on empty documents
    categories = {k: v for i, (k, v) in enumerate(categories.items()) if i not in empty_docs}
    taxonomies = {k: v for i, (k, v) in enumerate(taxonomies.items()) if i not in empty_docs}
    authors = {k: v for i, (k, v) in enumerate(authors.items()) if i not in empty_docs}
    texts = {k: v for i, (k, v) in enumerate(texts.items()) if i not in empty_docs}
    id2doc_file = {k: v for i, (k, v) in enumerate(id2doc_file.items()) if i not in empty_docs}

    # fix document IDs after removing empty documents
    texts = {i: v for i, v in zip(range(len(texts.values())), list(texts.values()))}
    id2doc_file = {i: v for i, v in zip(range(len(id2doc_file.values())), list(id2doc_file.values()))}

    doc2bow = make_doc2bow(corpora, documents)

    doc_word_matrix = sparse_vector_document_representations(corpora, doc2bow)

    cat2id, categories, auth2id, authors, tax2id, taxonomies = construct_metadata([categories, authors, taxonomies])
    
    doc_cat_one_hot = torch.zeros((len(documents), len(cat2id)))
    for i, doc in enumerate(documents):
        doc_cat_one_hot[i, categories.get(i)] = 1

    if save:
        if printouts:
            print("Saving Corpora & Preprocessed Text")
        corpora.save('../' + update_path(paths['corpora'], folder_name))
        with open('../' + update_path(paths['doc2bow'], folder_name), "wb") as file:
            pickle.dump(doc2bow, file)
        with open('../' + update_path(paths['doc_word_matrix'], folder_name), "wb") as file:
            pickle.dump(doc_word_matrix, file)
        with open('../' + paths['doc_cat_one_hot_matrix'], "wb") as file:
            pickle.dump(doc_cat_one_hot, file)
        utility.save_dict_file('../' + update_path(paths['id2doc'], folder_name), id2doc_file)
        utility.save_dict_file('../' + update_path(paths['doc2raw_text'], folder_name), texts)
        utility.save_dict_file('../' + update_path(paths['id2category'], folder_name), cat2id)
        utility.save_dict_file('../' + update_path(paths['id2author'], folder_name), auth2id)
        utility.save_dict_file('../' + update_path(paths['id2taxonomy'], folder_name), tax2id)
        utility.save_dict_file('../' + update_path(paths['doc2category'], folder_name), categories)
        utility.save_dict_file('../' + update_path(paths['doc2author'], folder_name), authors)
        utility.save_dict_file('../' + update_path(paths['doc2taxonomy'], folder_name), taxonomies)
        utility.save_dict_file('../' + update_path(paths['id2word'], folder_name), {v: k for k, v in corpora.token2id.items()})
        utility.save_dict_file('../' + update_path(paths['doc2pre_text'], folder_name), documents)
        utility.save_dict_file('../' + update_path(paths['doc2word'], folder_name), doc2id)

    if printouts:
        print('Preprocessing Finished.')
    return corpora, documents, doc2bow, doc_word_matrix

  
def construct_metadata(meta):
    categories, authors, taxonomies = meta

    # Filter meta data mappings:
    distribution = {}
    for v in categories.values():
        distribution[v] = distribution.get(v, 0) + 1
    bad_cat = [k for k, v in distribution.items() if v < 139]
    categories = {k: v if v not in bad_cat else 'misc' for k, v in categories.items()}

    distribution = {}
    for v in authors.values():
        distribution[v] = distribution.get(v, 0) + 1
    bad_cat = [k for k, v in distribution.items() if v < 14]
    authors = {k: v if v not in bad_cat else 'misc' for k, v in authors.items()}

    distribution = {}
    for v in taxonomies.values():
        for tax in v.split('/'):
            distribution[tax] = distribution.get(tax, 0) + 1
    bad_cat = [k for k, v in distribution.items() if v < 14]
    taxonomies = {k: '/'.join([x for x in v.split('/') if x not in bad_cat]) for k, v in taxonomies.items()}

    # make value -> id mappings
    cat2id = {v: i for v, i in zip(list(set(categories.values())), range(len(set(categories.values()))))}
    auth2id = {v: i for v, i in zip(list(set(authors.values())), range(len(set(authors.values()))))}
    tax2id = set([x for v in set(taxonomies.values()) for x in v.split('/')])
    tax2id = {v: k for k, v in enumerate(tax2id)}

    # make doc_id -> meta_id mappings
    categories = {i: cat2id[v] for v, i in zip(categories.values(), range(len(categories)))}
    authors = {i: auth2id[v] for v, i in zip(authors.values(), range(len(authors)))}
    taxonomies = {i: [tax2id[''] if v == '' else tax2id[x] for x in v.split('/')]
                  for v, i in zip(taxonomies.values(), range(len(taxonomies)))}

    cat2id = {v: k for k, v in cat2id.items()}
    auth2id = {v: k for k, v in auth2id.items()}
    tax2id = {v: k for k, v in tax2id.items()}

    return cat2id, categories, auth2id, authors, tax2id, taxonomies


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
    geographic_category_names = ["Frederikshavn-avis", "Morsø Debat", "Morsø-avis", "Rebild-avis", "Brønderslev-avis",
                                 "Thisted-avis", "Jammerbugt-avis", "Vesthimmerland-avis", "Hjørring-avis",
                                 "Aalborg-avis", "Morsø Sport", "Thisted sport", "Mariagerfjord-avis", "Udland-avis"]
    info = preprocessing(json_file='full_json', printouts=True, save=True, folder_name='full')
    print('Finished Preprocessing')
