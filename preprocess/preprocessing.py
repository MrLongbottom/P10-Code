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
    categories = {e: v for e, (k, v) in enumerate(categories.items()) if k not in bad_ids}
    authors = {e: v for e, (k, v) in enumerate(authors.items()) if k not in bad_ids}
    taxonomies = {e: v for e, (k, v) in enumerate(taxonomies.items()) if k not in bad_ids}
    texts = {e: v for e, (k, v) in enumerate(texts.items()) if k not in bad_ids}
    if save:
        if printouts:
            print("Saving data mapping files")
        utility.save_dict_file('../' + paths['id2doc_name'], id2doc_file)
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

    if save:
        if printouts:
            print("Saving Corpora & Preprocessed Text")
        corpora.save('../' + paths['corpora'])
        with open('../' + paths['doc2bow'], "wb") as file:
            pickle.dump(doc2bow, file)
        with open('../' + paths['doc_word_matrix'], "wb") as file:
            pickle.dump(doc_word_matrix, file)
        utility.save_dict_file('../' + paths['id2word'], {v: k for k, v in corpora.token2id.items()})
        utility.save_dict_file('../' + paths['id2pre_text'], documents, separator=':')
        utility.save_dict_file('../' + paths['doc2word_ids'], doc2id, separator=':')

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
    # TODO utility load
    # TODO utility load with separator
    # TODO corpora load
    # TODO pickle load
    raise NotImplementedError


if __name__ == '__main__':
    info = preprocessing(printouts=True, save=True)
    print('Finished Preprocessing')
