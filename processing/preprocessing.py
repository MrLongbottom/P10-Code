import gensim
from tqdm import tqdm
import itertools
from processing import loading
import torch


def preprocessing(printouts=False, save=False):
    # load data file
    if printouts:
        print("Loading dataset")
    texts, categories, authors, taxonomies = loading.load_document_file("../data/2017_data.json")

    # removing duplicates from dictionaries
    rev = {v: k for k, v in texts.items()}
    new_texts = {v: k for k, v in rev.items()}
    bad_ids = [x for x in texts.keys() if x not in new_texts.keys()]
    categories = {k: v for k, v in categories.items() if k not in bad_ids}
    authors = {k: v for k, v in authors.items() if k not in bad_ids}
    taxonomies = {k: v for k, v in taxonomies.items() if k not in bad_ids}
    texts = new_texts
    # TODO save down doc_id -> metadata mappings

    # tokenize (document token generators)
    if printouts:
        print("Tokenization")
    documents = [list(gensim.utils.tokenize(x, lowercase=True, deacc=True, encoding='utf-8')) for x in
                 tqdm(texts.values())]

    # normalize (currently doesn't work)
    # documents = [[y for y in gensim.utils.tokenize(x[0], lowercase=True, deacc=True, encoding='utf-8')] for x in data.values()]
    # norm_documents = gensim.models.normmodel.NormModel(corpus=[corpora.doc2bow(x) for x in documents], norm='l1')
    # norm_corpora = gensim.corpora.Dictionary.from_corpus(norm_documents)

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
            print("Saving Corpora")
        corpora.save("generated_files/corpora")
    if printouts:
        print('Preprocessing Finished.')
    return corpora, documents


def sparse_vector_document_representations(corpora, documents):
    doc2bow = []
    for doc in documents:
        doc2bow.append(corpora.doc2bow(doc))
    doc_keys = list(itertools.chain.from_iterable
                    ([[[doc_id, word_id[0]] for word_id in doc2bow[doc_id]] for doc_id in range(len(doc2bow))]))
    doc_values = []
    for doc in tqdm(doc2bow):
        [doc_values.append(y) for x, y in doc]
    sparse_docs = torch.sparse.FloatTensor(torch.LongTensor(doc_keys).t(), torch.FloatTensor(doc_values),
                                          torch.Size([corpora.num_docs, len(corpora)]))
    return sparse_docs


if __name__ == '__main__':
    corpora = preprocessing(printouts=True)
